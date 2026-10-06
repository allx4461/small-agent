from pathlib import Path
import os
import yaml
from dataclasses import dataclass

# Model and router settings
BASE_TEMPERATURE = 0.5
ALLOWED_DELTAS = frozenset({-1, 0, 1})
_MAX_PARSE_ATTEMPTS = 4

PROMPTS_PATH = Path(__file__).resolve().with_name("agent_prompts.yaml")
_PROMPTS = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))

ROUTER_SYSTEM_PROMPT = _PROMPTS["router"]["system"]
ROUTER_REPAIR_PROMPT = _PROMPTS["router"]["repair"]
TEMPERATURE_STEP = 0.2
RAG_MIN_SIMILARITY = 0.4

TOOL_DESCRIPTIONS = {
    "nice_answer": """Clean up the whitespace around lines in a text while preserving its content.
Args:
    text: The text whose line whitespace should be cleaned.
Returns:
    The text with whitespace trimmed from each line.""",
    "transliterate_to_english": """Transliterate Russian text into Latin characters only when the user explicitly asks for transliteration.
Args:
    text: Russian text to transliterate.
Returns:
    The transliterated text.""",
}

BAD_TOOL_DESCRIPTIONS = {
    "nice_answer": """Makes text nicer.
Args:
    text: Some text.
Returns:
    Text.
""",
    "transliterate_to_english": """Transliterates Russian text into Latin characters.
Args:
    text: Russian text to transliterate.
Returns:
    The transliterated text.
""",
}
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = PROJECT_DIR / "eval_cases.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "evaluation_results.json"
DEFAULT_EVALUATOR_MODEL = os.environ.get(
    "EVAL_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"
)

JUDGE_PROMPT = """You are a strict evaluator for a small RAG agent.
Score only the supplied observable evidence. Do not infer hidden reasoning.
The action trace contains tool names/errors/finalization metadata, not chain-of-thought.
Return one JSON object only:
{
  "route_score": integer 0-2,
  "retrieval_score": integer 0-2,
  "action_score": integer 0-2,
  "answer_score": integer 0-4,
  "passed": boolean,
  "rationale": short string
}
Scoring:
- route_score: whether RAG routing agrees with expected_needs_rag (2 exact, 0 wrong).
- retrieval_score: follow expected_retrieval. "relevant" means the context supports the case evidence (2 sufficient, 1 partial, 0 absent/wrong); "none" means no context was retrieved (2) and the answer appropriately abstains. For non-RAG cases, score 2 if no retrieval was done, otherwise 0.
- action_score: whether observed actions match expected_tool (2 correct, 1 acceptable but unnecessary/partial, 0 incorrect or tool error). If expected_tool is null, score 2 when there are no failed actions and the run finalized.
- answer_score: factuality, relevance, and whether expected_facts are supported by the supplied context (4 correct and grounded, 2 partially correct, 0 wrong/unsupported).
- passed is true only when route_score=2, retrieval_score=2, action_score=2 and answer_score >= 3.
Be strict about unsupported factual claims. Explain any failed dimension briefly.
"""


@dataclass(frozen=True)
class PromptProfile:
    name: str
    agent_system_prompt: str
    router_system_prompt: str
    router_repair_prompt: str
    tool_descriptions: dict[str, str]


def get_prompt_profile(name: str) -> PromptProfile:
    """Return one complete prompt profile or explain which bad prompts are missing."""
    if name == "good":
        return PromptProfile(
            name=name,
            agent_system_prompt=_PROMPTS["system_prompt"],
            router_system_prompt=ROUTER_SYSTEM_PROMPT,
            router_repair_prompt=ROUTER_REPAIR_PROMPT,
            tool_descriptions=TOOL_DESCRIPTIONS,
        )
    if name != "bad":
        raise ValueError(f"Unknown prompt profile: {name!r}; expected 'good' or 'bad'")

    bad_prompts = _PROMPTS.get("bad_profile", {})
    values = {
        "agent_system_prompt": bad_prompts.get("agent_system_prompt"),
        "router_system_prompt": bad_prompts.get("router_system_prompt"),
        "router_repair_prompt": bad_prompts.get("router_repair_prompt"),
        **{f"tool_descriptions.{key}": value for key, value in BAD_TOOL_DESCRIPTIONS.items()},
    }
    missing = [key for key, value in values.items() if not isinstance(value, str) or not value.strip()]
    if missing:
        raise ValueError(
            "The bad prompt profile is not complete yet. Add these prompts before "
            f"comparing profiles: {', '.join(missing)}"
        )

    return PromptProfile(
        name=name,
        agent_system_prompt=values["agent_system_prompt"],
        router_system_prompt=values["router_system_prompt"],
        router_repair_prompt=values["router_repair_prompt"],
        tool_descriptions={
            key: BAD_TOOL_DESCRIPTIONS[key] for key in BAD_TOOL_DESCRIPTIONS
        },
    )
