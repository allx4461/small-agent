from pathlib import Path

import yaml

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
