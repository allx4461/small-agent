# region Imports
from transliterate import translit
from smolagents import CodeAgent, FinalAnswerTool, InferenceClientModel, OpenAIServerModel, load_tool, tool, GradioUI
import requests
import pytz
import os
import yaml
from router import BASE_TEMPERATURE, RouterDecision, route_query
from minirag import rag_search as minirag_search
# endregion


# region Agent tools
@tool
def nice_answer(text:str)->str:
    """a tool that makes the answer more nice and readable.
    Args:
        text (str): The answer text to be improved.
    Returns:
        str: The improved and more readable answer text.
    """
    return '\n'.join([line.strip() for line in text.splitlines()])
@tool
def transliterate_to_english(text:str)->str:
    """do not use this tool if user does not directly ask for transliteration.
    a tool that transliterates text from Russian to English. 
    example: фузз-> fuzz
    Args:
        text (str): The text in Russian to be transliterated to English.
    Returns:
        str: The transliterated text in English.
    """
    return translit(text, reversed=True)
@tool
def rag_search(query: str) -> str:
    """Searches the internal knowledge base for documentation. Use when user asks about concepts, definitions, or how things work.
    Args:
        query (str): The search query.
    Returns:
        str: The search results from the internal knowledge base.
    """
    return minirag_search(query)

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
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)


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
TEMPERATURE_STEP = 0.2


def answer(query: str) -> str:
    """Router first, then the agent: the router picks the temperature, RAG flag and cleaned query."""
    # 1. Роутер (router.py) возвращает RouterDecision
    try:
        decision = route_query(query, router_model)
    except ValueError as exc:
        # Router failed: neutral decision and the ORIGINAL query, nothing is lost.
        print(f"[router] fallback, reason: {exc}")
        decision = RouterDecision(needs_rag=False, temperature_delta=0, rewritten_query=query)
    print(f"[router] original={query!r}\n[router] decision={decision}")

    # 2. temperature_delta -> температура ЭТОГО запроса (ограничена 0..1)
    temperature = min(max(BASE_TEMPERATURE + decision.temperature_delta * TEMPERATURE_STEP, 0.0), 1.0)

    # 3. needs_rag: поиск выполняется в коде, найденное кладётся в задачу агента
    task = decision.rewritten_query
    if decision.needs_rag:
        docs = minirag_search(decision.rewritten_query)
        print(f"[rag] {docs!r}")
        task = f"{task}\n\nКонтекст из базы знаний (отвечай по нему, не выдумывай факты):\n{docs}"

    # 4. Агент создаётся уже с нужной температурой и итоговой задачей
    return build_agent(temperature).run(task)
# endregion


# region Entry point
if __name__ == "__main__":
    query = "Что такое температура в контексте LLM и как она влияет на ответы? Подними температуру на 0.2. Какоая компания пользуется GigaChat согласно документации?"
    print(f"Query: {query}")
    print("--- ЗАПУСК АГЕНТА ---")
    result = answer(query)
    print("\n--- ИТОГОВЫЙ ОТВЕТ АГЕНТА ---")
    print(result)
# endregion
