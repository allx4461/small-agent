# region Imports
from transliterate import translit
from smolagents import CodeAgent, FinalAnswerTool, InferenceClientModel, OpenAIServerModel, tool
import os
import yaml
from config import BASE_TEMPERATURE, TEMPERATURE_STEP, TOOL_DESCRIPTIONS
from router import RouterDecision, route_query
from minirag import rag_search as minirag_search
from pathlib import Path
# endregion


# region Agent tools
def nice_answer(text:str)->str:
    return '\n'.join([line.strip() for line in text.splitlines()])


def transliterate_to_english(text:str)->str:
    return translit(text, reversed=True)


def _register_configured_tool(function, name: str):
    function.__doc__ = TOOL_DESCRIPTIONS[name]#красивенько достаем описание из конфига
    return tool(function)#и явно переводим в тул


nice_answer = _register_configured_tool(nice_answer, "nice_answer")
transliterate_to_english = _register_configured_tool(
    transliterate_to_english, "transliterate_to_english"
)

final_answer = FinalAnswerTool()  # обязательный инструмент: им агент завершает работу
# endregion


# region Models
# Провайдер задаётся переменными окружения (любой OpenAI-совместимый API):
#   LLM_BASE_URL  - адрес API, LLM_API_KEY - ключ, LLM_MODEL - модель агента,
#   ROUTER_MODEL  - модель роутера (необязательно, по умолчанию как у агента).
# Если LLM_BASE_URL не задан, используется Hugging Face как раньше.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", LLM_MODEL)


def make_model(model_id: str, temperature: float = BASE_TEMPERATURE):
    if LLM_BASE_URL:
        return OpenAIServerModel(
            model_id=model_id,
            api_base=LLM_BASE_URL,
            api_key=os.environ.get("LLM_API_KEY", "not-needed"),
            max_tokens=2096,
            temperature=temperature,
        )
    return InferenceClientModel(
        max_tokens=2096,
        temperature=temperature,
        model_id=model_id,
        custom_role_conversions=None,
    )


# Роутер не должен быть "творческим": нулевая температура даёт повторяемые решения.
router_model = make_model(ROUTER_MODEL, temperature=0.0)
# endregion


# region Agent assembly
# CodeAgent expects only its own template keys; router prompts are separate.
PROMPTS_PATH = Path(__file__).resolve().with_name("agent_prompts.yaml")
with PROMPTS_PATH.open(encoding="utf-8") as stream:
    all_prompts = yaml.safe_load(stream)

prompt_templates = {
    key: all_prompts[key]
    for key in ("system_prompt", "planning", "managed_agent", "final_answer")
}


def build_agent(temperature: float) -> CodeAgent:
    """Новый агент и новая модель на каждый запрос: у агента своя память шагов,
    а температура не лежит в общем объекте, который меняют параллельные запросы.
    rag_search агенту не дан: решение о RAG принадлежит только роутеру.
    """
    return CodeAgent(
        model=make_model(LLM_MODEL, temperature=temperature),
        tools=[final_answer, nice_answer, transliterate_to_english],
        max_steps=6,
        verbosity_level=2,
        planning_interval=None,
        name=None,
        description=None,
        prompt_templates=prompt_templates,
    )
# endregion


# region Router -> Agent


def _format_history(history: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"Пользователь: {user_message}\nАссистент: {assistant_message}"
        for user_message, assistant_message in history
    )


def answer(query: str, history: list[tuple[str, str]] | None = None) -> str:
    """Run one turn, using earlier visible turns as context when provided."""
    history = history or []
    conversation_context = _format_history(history)

    # 1. Роутер (router.py) возвращает RouterDecision
    try:
        decision = route_query(
            query,
            router_model,
            conversation_context=conversation_context or None,
        )
    except ValueError as exc:
        # Router failed: neutral decision and the ORIGINAL query, nothing is lost.
        print(f"[router] fallback, reason: {exc}")
        decision = RouterDecision(needs_rag=False, temperature_delta=0, rewritten_query=query)
    print(f"[router] original={query!r}\n[router] decision={decision}")

    # 2. temperature_delta -> температура ЭТОГО запроса (ограничена 0..1)
    temperature = min(max(BASE_TEMPERATURE + decision.temperature_delta * TEMPERATURE_STEP, 0.0), 1.0)

    # 3. needs_rag: поиск выполняется в коде, найденное кладётся в задачу агента
    task = (
        f"История предыдущего диалога:\n{conversation_context}\n\n"
        f"Текущий запрос пользователя:\n{query}\n\n"
        f"Суть текущего запроса:\n{decision.rewritten_query}"
        if conversation_context
        else decision.rewritten_query
    )
    if decision.needs_rag:
        docs = minirag_search(decision.rewritten_query)
        if docs is None:
            print("[rag] no sufficiently relevant document found")
            task = (
                f"{task}\n\nПоиск не нашёл достаточно релевантного контекста. "
                "Сообщи об этом и не выдумывай ответ от имени документации."
            )
        else:
            print(f"[rag] {docs!r}")
            task = f"{task}\n\nКонтекст из базы знаний (отвечай по нему, не выдумывай факты):\n{docs}"

    # 4. Агент создаётся уже с нужной температурой и итоговой задачей
    return build_agent(temperature).run(task)
# endregion


# region Entry point
def run_cli() -> None:
    """Run a single-process chat session; its visible turns live only in RAM."""
    history: list[tuple[str, str]] = []
    print("Чат запущен. Команды: /clear — очистить историю, /exit — выйти.")

    while True:
        try:
            query = input("Вы: ").strip()
        except EOFError:
            break

        if query.lower() in {"/exit", "/quit"}:
            break
        if query == "/clear":
            history.clear()
            print("История очищена.")
            continue
        if not query:
            continue

        response = answer(query, history)
        history.append((query, response))
        print(f"Агент: {response}")


if __name__ == "__main__":
    run_cli()
# endregion
