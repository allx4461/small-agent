import json
import re
from dataclasses import dataclass
from typing import Any
from smolagents import InferenceClientModel
from config import (
    ALLOWED_DELTAS,
    _MAX_PARSE_ATTEMPTS,
    ROUTER_REPAIR_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)


@dataclass(frozen=True)
class RouterDecision:
    needs_rag: bool
    temperature_delta: int
    rewritten_query: str

#region helper-functions
def _extract_json_candidate(text: str) -> str:
    """убирает лишние артефакты для корректного перевода в json"""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"\{[\s\S]*\}", stripped)
    if match:
        return match.group(0)
    return stripped


def _validate_router_payload(data: dict[str, Any]) -> RouterDecision:
    """проверка переданного json и перевод в датакласс Router Decision
    """
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not a dictionary")

    expected_keys = {"needs_rag", "temperature_delta", "rewritten_query"}
    keys = set(data.keys())
    if keys != expected_keys:
        raise ValueError(f"Parsed JSON keys are not as expected: got {sorted(keys)}")

    if not isinstance(data["needs_rag"], bool):
        raise ValueError("needs_rag must be a boolean")

    temperature_delta = data["temperature_delta"]
    if type(temperature_delta) is not int or temperature_delta not in ALLOWED_DELTAS:
        raise ValueError("temperature_delta must be one of -1, 0, 1")

    rewritten = data["rewritten_query"]
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError("rewritten_query must be a non-empty string")

    return RouterDecision(
        needs_rag=data["needs_rag"],
        temperature_delta=temperature_delta,
        rewritten_query=rewritten.strip(),
    )


def _coerce_model_text(response: Any) -> str:
    """обрабатывает вывод модели, делая все строкой"""
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    raise TypeError("Router model returned unsupported response type")


def _repair_router_output(router_model, broken_output: str, error_text: str) -> str:
    """если рутер вернул некорректный вывод, даем инфо об ошибке и просим исправить"""
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ROUTER_REPAIR_PROMPT.replace(
                "{{broken_output}}", broken_output
            ).replace("{{error_text}}", error_text),
        },
    ]
    fixed = router_model(messages=messages)
    return _coerce_model_text(fixed)
#endregion
#region model functions
def parse_router_decision(json_str: str, router_model=None) -> RouterDecision:
    """Parse router decision; if parsing fails, optionally ask model to repair output.

    Loop detection stops retries when the model keeps repeating the same invalid payload 
    """
    candidate = json_str
    seen_invalid_outputs: set[str] = set()

    for _ in range(_MAX_PARSE_ATTEMPTS):
        normalized = _extract_json_candidate(candidate)
        try:
            data = json.loads(normalized)
            return _validate_router_payload(data)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            if router_model is None:
                raise ValueError(f"Invalid router output: {candidate}") from exc

            if normalized in seen_invalid_outputs:
                raise ValueError(
                    "Router output repair loop detected: model repeated the same invalid payload"
                ) from exc
            seen_invalid_outputs.add(normalized)

            candidate = _repair_router_output(router_model, candidate, str(exc))

    raise ValueError("Failed to parse router decision after repair attempts")


def route_query(
    query: str,
    router_model,
    conversation_context: str | None = None,
) -> RouterDecision:
    """из запроса формирует решение по температуре, rag, и переписанному запросу."""
    user_message = query
    if conversation_context:
        user_message = (
            "Use the prior conversation only to resolve references in the current message. "
            "Make the routing decision for the current message, not old requests.\n\n"
            f"Prior conversation:\n{conversation_context}\n\n"
            f"Current user message:\n{query}"
        )
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    decision = parse_router_decision(
        _coerce_model_text(router_model(messages)),
        router_model=router_model,
    )
    if len(decision.rewritten_query) > 2 * len(query) + 50:
        raise ValueError("rewritten_query is much longer than the original query")
    return decision

#endregion