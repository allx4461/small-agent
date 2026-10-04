# region Imports
from transliterate import translit
from smolagents import CodeAgent, FinalAnswerTool, InferenceClientModel, load_tool, tool, GradioUI
import requests
import pytz
import yaml
from router import BASE_TEMPERATURE, RouterDecision, route_query
from minirag import rag_search
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
    """a tool that transliterates text from Russian to English. use when user intends to convert Russian text to English transliteration.
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
    return rag_search(query)

final_answer = FinalAnswerTool()  # обязательный инструмент: им агент завершает работу
# endregion


# region Models
router_model = InferenceClientModel(
    max_tokens=2096,
    temperature=BASE_TEMPERATURE,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)
model = InferenceClientModel(
    max_tokens=2096,
    temperature=BASE_TEMPERATURE,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)
# endregion


# region Agent assembly
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[final_answer, nice_answer, transliterate_to_english, rag_search],
    max_steps=6,
    verbosity_level=2,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)
# endregion


# region Router -> Agent 
TEMPERATURE_STEP = 0.2


def answer(query: str) -> str:
    """Router first, then the agent: the router picks the temperature and RAG flag."""
    # 1. Роутер (router.py) спрашивает модель и возвращает RouterDecision
    try:
        decision = route_query(query, router_model)
    except ValueError as exc:
        # Router failed to produce valid JSON: fall back to the neutral decision with the original query.
        print(f"[router] fallback, reason: {exc}")
        decision = RouterDecision(needs_rag=False, temperature_delta=0, rewritten_query=query)

    # 2. temperature_delta -> реальная температура в общей модели (это и есть передача данных агенту)
    temperature = BASE_TEMPERATURE + decision.temperature_delta * TEMPERATURE_STEP
    rewritten_query = decision.rewritten_query
    model.kwargs["temperature"] = min(max(temperature, 0.0), 1.0)
    print(f"[router] {decision}, temperature={model.kwargs['temperature']}")

    # 3. needs_rag пока только печатается, агенту НЕ передаётся
    if decision.needs_rag:
        # RAG is not implemented yet: hook retrieval in here and pass the docs into the task.
        print("[router] needs_rag=True, but RAG is not connected yet")

    # 4. Агент стартует уже с обновлённой температурой и запросом без упоминаний инструментов
    return agent.run(rewritten_query)
# endregion


# region Entry point
if __name__ == "__main__":
    query = "Что такое температура в контексте LLM и как она влияет на ответы? Подними температуру на 0.2"
    print(f"Query: {query}")
    print("--- ЗАПУСК АГЕНТА ---")
    result = answer(query)
    print("\n--- ИТОГОВЫЙ ОТВЕТ АГЕНТА ---")
    print(result)
# endregion
