import json
import re
from dataclasses import dataclass
from typing import Any
from smolagents import InferenceClientModel

BASE_TEMPERATURE = 0.5
ALLOWED_DELTAS = frozenset({-1, 0, 1})
_MAX_PARSE_ATTEMPTS = 4


@dataclass(frozen=True)
class RouterDecision:
    needs_rag: bool
    temperature_delta: int
    rewritten_query: str


ROUTER_SYSTEM_PROMPT = """you are a router that decides whether to use RAG and what temperature delta to apply (if user asks to).
return only a JSON object with exactly these keys: {"needs_rag": boolean, "temperature_delta": integer, "rewritten_query": string}
1. use RAG only if external documentation is needed
2. transliteration does not need to use RAG, there is another tool for that
3. receiving ambiguous queries = return exactly `{"needs_rag": false, "temperature_delta": 0, "rewritten_query": "<original_query>"}`
4. rewrite the query: remove only instructions addressed to your layer (e.g. "make the answer more creative/precise" that you turned into temperature_delta). Keep the topic and the actual question unchanged, even if the topic itself is about temperature.
5. if user asks to raise temperature, increase temperature_delta by 1; if user asks to lower temperature, decrease temperature_delta by 1. Do not change temperature_delta for other instructions. """


def _extract_json_candidate(text: str) -> str:
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
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not a dictionary")

    expected_keys = {"needs_rag", "temperature_delta", "rewritten_query"}
    keys = set(data.keys())
    if keys != expected_keys:
        raise ValueError(f"Parsed JSON keys are not as expected: got {sorted(keys)}")

    if not isinstance(data["needs_rag"], bool):
        raise ValueError("needs_rag must be a boolean")

    temperature_delta = data["temperature_delta"]
    if not isinstance(temperature_delta, int) or temperature_delta not in ALLOWED_DELTAS:
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
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    raise TypeError("Router model returned unsupported response type")


def _repair_router_output(router_model, broken_output: str, error_text: str) -> str:
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Your previous router output is invalid. Fix it and return only valid JSON "
                "with keys needs_rag (bool), temperature_delta (int in -1,0,1) and rewritten_query (non-empty string).\n"
                f"Invalid output:\n{broken_output}\n"
                f"Validation error: {error_text}"
            ),
        },
    ]
    fixed = router_model(messages=messages)
    return _coerce_model_text(fixed)


def parse_router_decision(json_str: str, router_model=None) -> RouterDecision:
    """Parse router decision; if parsing fails, optionally ask model to repair output.

    Loop detection stops retries when the model keeps repeating the same invalid
    payload (common failure mode when instruction following degrades).
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


def route_query(query: str, router_model) -> RouterDecision:
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = router_model(messages)
    result_text = _coerce_model_text(response)
    return parse_router_decision(result_text, router_model=router_model)

router_model = InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)
