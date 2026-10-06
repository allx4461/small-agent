"""Run the agent against labeled cases and score answers with an LLM judge.

The judge sees final answers and observable actions only. Hidden reasoning and
generated Python code are intentionally excluded from evaluation input/output.
"""

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from agent import answer_with_trace, make_model
from config import (
    DEFAULT_CASES,
    DEFAULT_EVALUATOR_MODEL,
    DEFAULT_OUTPUT,
    JUDGE_PROMPT,
    PROJECT_DIR,
    get_prompt_profile,
)

#region utils
def _read_cases(path: Path) -> list[dict]:
    """прочитать кейсы из .jsonl и вернуть их в виде списка словарей."""
    cases = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            required = {
                "id", "query", "expected_needs_rag", "expected_retrieval",
                "expected_facts", "expected_tool",
            }
            if not required.issubset(case):
                raise ValueError(f"Missing case fields at {path}:{line_number}")
            if type(case["expected_needs_rag"]) is not bool:
                raise ValueError(f"expected_needs_rag must be boolean at {path}:{line_number}")
            if case["expected_retrieval"] not in {"relevant", "none"}:
                raise ValueError(f"Invalid expected_retrieval at {path}:{line_number}")
            cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def _extract_json(text: str) -> dict:
    """извлечь JSON-объект из текста, возвращаемого судьей."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Judge did not return a JSON object")
    result = json.loads(match.group())
    if not isinstance(result, dict):
        raise ValueError("Judge response must be a JSON object")
    score_fields = ("route_score", "retrieval_score", "action_score", "answer_score")
    for field in score_fields:
        if type(result.get(field)) is not int:
            raise ValueError(f"Judge field {field} must be an integer")
    if not 0 <= result["route_score"] <= 2:
        raise ValueError("route_score must be between 0 and 2")
    if not 0 <= result["retrieval_score"] <= 2:
        raise ValueError("retrieval_score must be between 0 and 2")
    if not 0 <= result["action_score"] <= 2:
        raise ValueError("action_score must be between 0 and 2")
    if not 0 <= result["answer_score"] <= 4:
        raise ValueError("answer_score must be between 0 and 4")
    if type(result.get("passed")) is not bool or not isinstance(result.get("rationale"), str):
        raise ValueError("Judge response requires boolean passed and string rationale")
    # Derive pass/fail from validated scores rather than trusting a contradictory
    # free-form verdict from the judge model.
    result["passed"] = (
        result["route_score"] == 2
        and result["retrieval_score"] == 2
        and result["action_score"] == 2
        and result["answer_score"] >= 3
    )
    return result


def _judge_case(judge_model, case: dict, observation: dict) -> dict:
    """оценить кейс с помощью ллм, вернуть словарь с полями route_score, retrieval_score, action_score, answer_score, passed и rationale."""
    payload = {"case": case, "observed": observation}
    response = judge_model(messages=[
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    content = response if isinstance(response, str) else getattr(response, "content", None)
    if not isinstance(content, str):
        raise TypeError("Evaluation model returned no textual content")
    return _extract_json(content)
#endregion
#region evaluate
def evaluate(
    cases_path: Path = DEFAULT_CASES,
    output_path: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
    evaluator_model_id: str = DEFAULT_EVALUATOR_MODEL,
    profiles: tuple[str, ...] = ("good", "bad"),
) -> dict:
    """выставить оценки для всех кейсов с использованием указанного оценщика и профилей промптов."""
    cases = _read_cases(cases_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        cases = cases[:limit]

    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("Select one or more unique prompt profiles")
    # Validate all profiles before paying for any model requests.
    for profile_name in profiles:
        get_prompt_profile(profile_name)

    judge_model = make_model(evaluator_model_id, temperature=0.0)
    results = []
    for case_index, case in enumerate(cases, 1):
        for profile_name in profiles:
            print(
                f"[{case_index}/{len(cases)}] {case['id']} "
                f"(profile={profile_name}): running agent"
            )
            turn = answer_with_trace(
                case["query"],
                case.get("history"),
                prompt_profile=profile_name,
            )
            observation = {
                "prompt_profile": profile_name,
                "route": asdict(turn.route),
                "retrieved_context": turn.retrieved_context,
                "action_trace": turn.action_trace,
                "agent_state": turn.agent_state,
                "answer": turn.answer,
                "duration_seconds": round(turn.duration_seconds, 2),
            }
            judgment = _judge_case(judge_model, case, observation)
            results.append({
                "id": case["id"],
                "prompt_profile": profile_name,
                "query": case["query"],
                "observed": observation,
                "judgment": judgment,
            })
            profile_results = [
                row for row in results if row["prompt_profile"] == profile_name
            ]
            report = {
                "evaluator_model": evaluator_model_id,
                "case_count_per_profile": len(profile_results),
                "profiles": {
                    name: {
                        "case_count": sum(row["prompt_profile"] == name for row in results),
                        "passed_count": sum(
                            row["judgment"]["passed"]
                            for row in results
                            if row["prompt_profile"] == name
                        ),
                    }
                    for name in profiles
                },
                "results": results,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"  passed={judgment['passed']} "
                f"route={judgment['route_score']}/2 "
                f"retrieval={judgment['retrieval_score']}/2 "
                f"actions={judgment['action_score']}/2 "
                f"answer={judgment['answer_score']}/4"
            )

    print(f"Evaluation complete: {report['profiles']}. Report: {output_path}")
    return report
#endregion
#region entrypoint
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--profiles",
        default="good,bad",
        help="Comma-separated prompt profiles to compare (default: good,bad)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EVALUATOR_MODEL,
        help="LLM judge model (provider configuration is shared with the agent)",
    )
    args = parser.parse_args()
    selected_profiles = tuple(name.strip() for name in args.profiles.split(",") if name.strip())
    evaluate(args.cases, args.output, args.limit, args.model, selected_profiles)


if __name__ == "__main__":
    main()
#endregion
