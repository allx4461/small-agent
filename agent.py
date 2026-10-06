# region Imports
from transliterate import translit
from smolagents import CodeAgent, FinalAnswerTool, InferenceClientModel, OpenAIServerModel, tool
import time
import os
import yaml
from dataclasses import dataclass
from types import FunctionType
from config import BASE_TEMPERATURE, TEMPERATURE_STEP, get_prompt_profile
from router import RouterDecision, route_query
from minirag import rag_search as minirag_search
from pathlib import Path
# endregion


# region Agent tools
def nice_answer(text:str)->str:
    return '\n'.join([line.strip() for line in text.splitlines()])


def transliterate_to_english(text:str)->str:
    return translit(text, reversed=True)


def _register_configured_tool(function, description: str):
    configured_function = FunctionType(
        function.__code__,
        function.__globals__,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    configured_function.__annotations__ = function.__annotations__.copy()
    configured_function.__kwdefaults__ = function.__kwdefaults__
    configured_function.__doc__ = description
    return tool(configured_function)
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
    '''вернуть дефолтную инференс-модель для указанной модели и температуры'''
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


router_model = make_model(ROUTER_MODEL, temperature=0.0)#мне это не очень нравится, но рутер должен создаваться один раз вне функции
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


def build_agent(temperature: float, prompt_profile: str = "good") -> CodeAgent:
    """Новый агент и новая модель на каждый запрос: у агента своя память шагов,
    а температура не лежит в общем объекте, который меняют параллельные запросы.
    rag_search агенту не дан: решение о RAG принадлежит только роутеру.
    """
    profile = get_prompt_profile(prompt_profile)
    profile_templates = dict(prompt_templates)
    profile_templates["system_prompt"] = profile.agent_system_prompt
    tools = [
        FinalAnswerTool(),
        _register_configured_tool(nice_answer, profile.tool_descriptions["nice_answer"]),
        _register_configured_tool(
            transliterate_to_english,
            profile.tool_descriptions["transliterate_to_english"],
        ),
    ]
    return CodeAgent(
        model=make_model(LLM_MODEL, temperature=temperature),
        tools=tools,
        max_steps=6,
        verbosity_level=2,
        planning_interval=None,
        name=None,
        description=None,
        prompt_templates=profile_templates,
    )
# endregion


# region Router -> Agent

@dataclass
class TurnResult:
    answer: str #финальный ответ
    route: RouterDecision #решение рутера
    retrieved_context: str | None #извлечённый контекст, если был выполнен поиск RAG
    action_trace: list[dict[str, object]] #метаданные действий агента без скрытого рассуждения и сгенерированного кода
    agent_state: str #состояние агента в конце хода
    duration_seconds: float #время выполнения хода в секундах


def _observable_action_trace(run_steps: list[dict]) -> list[dict[str, object]]:
    """извлечь из шагов названия инструментов, ошибки и факт финального ответа"""
    trace = []
    for step in run_steps:
        calls = step.get("tool_calls") or []
        tool_names = []
        for call in calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                tool_names.append(str(name))

        if tool_names or step.get("error") or step.get("is_final_answer"):
            trace.append({
                "step_number": step.get("step_number"),
                "tools": tool_names,
                "had_error": bool(step.get("error")),
                "is_final_answer": bool(step.get("is_final_answer")),
            })
    return trace


def _format_history(history: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"user: {user_message}\nassistant: {assistant_message}"
        for user_message, assistant_message in history
    )


def answer_with_trace(
    query: str,
    history: list[tuple[str, str]] | None = None,
    prompt_profile: str = "good",
) -> TurnResult:
    """Run one turn and return the answer with its observable evaluation trace."""
    started_at = time.perf_counter()
    history = history or []
    conversation_context = _format_history(history)
    profile = get_prompt_profile(prompt_profile)

    # 1. Роутер (router.py) возвращает RouterDecision
    try:
        decision = route_query(
            query,
            router_model,
            conversation_context=conversation_context or None,
            system_prompt=profile.router_system_prompt,
            repair_prompt=profile.router_repair_prompt,
        )
    except ValueError as exc:
        # Router failed: neutral decision and the ORIGINAL query, nothing is lost.
        print(f"[router] fallback, reason: {exc}")
        decision = RouterDecision(needs_rag=False, temperature_delta=0, rewritten_query=query)
    print(f"[router] original={query!r}\n[router] decision={decision}")

    # 2. temperature_delta -> температура ЭТОГО запроса (ограничена 0..1)
    temperature = min(max(BASE_TEMPERATURE + decision.temperature_delta * TEMPERATURE_STEP, 0.0), 1.0)

    # 3. needs_rag: поиск выполняется в коде, найденное кладётся в задачу агента
    retrieved_context = None
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
            retrieved_context = docs
            print(f"[rag] {docs!r}")
            task = f"{task}\n\nКонтекст из базы знаний (отвечай по нему, не выдумывай факты):\n{docs}"

    # Keep only observable actions; do not persist model thoughts or generated code.
    run_result = build_agent(temperature, prompt_profile).run(
        task,
        return_full_result=True,
    )
    return TurnResult(
        answer=str(run_result.output),
        route=decision,
        retrieved_context=retrieved_context,
        action_trace=_observable_action_trace(run_result.steps or []),
        agent_state=run_result.state,
        duration_seconds=time.perf_counter() - started_at,
    )


def answer(query: str, history: list[tuple[str, str]] | None = None) -> str:
    """вернуть только финальный ответ агента."""
    return answer_with_trace(query, history).answer
# endregion


# region Entry point
def run_cli() -> None:
    """Запустить CLI для взаимодействия с агентом."""
    history: list[tuple[str, str]] = []
    print("Чат запущен. Команды: /clear — очистить историю, /exit — выйти.")

    while True:
        try:
            query = input("\nВаш запрос: ").strip()
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
        print(f"\nАгент: {response}")


if __name__ == "__main__":
    run_cli()
# endregion
