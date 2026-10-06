# Small Agent

Личный проект-эксперимент. Рутированный LLM-агент на smolagents с RAG по мини-базе знаний.
Цель - выяснить, как выбор модели и системного промта повлияет на перфоманс агента (при запросе к агенту по умолчанию используется "хороший" профиль, в evaluation сравниваются "хороший" и "плохой"),
чтобы понимать как построить наиболее эффективного мета-агента.

## Воркфлоу

```text
запрос + память этой сессии
                 |
                 v
             LLM рутер
                 |
    принятое решение (RouterDecision)
        /       |       \
  RAG     temperature  обработанный запрос
        \       |       /
          CodeAgent
              |
        итоговый ответ
```


## Что в каждом файле

| File | Responsibility |
| --- | --- |
| `agent.py` | CLI, создание агента,тулы, оркестрация |
| `router.py` | исходный запрос->регулировка параметров агента, переписанный запрос |
| `minirag.py` | RAG на мини-базе знаний |
| `config.py` | настройки, описания тулов |
| `agent_prompts.yaml` | промты для агента, инструкции для рутера |
| `eval_cases.jsonl` | 10 сценариев для проверки |
| `evaluation.py` | LLM-судья для оценки по метрикам |
| `tests/` | тесты для двух модулей |

## Требования

Python 3.12 is the development environment used for this prototype.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Прим.: по дефолту модели предоставляются из HF (используйте переменную окружения `HF_TOKEN`),
но можно настроить OpenAI-совместимые модели, вот так:

```bash
export LLM_BASE_URL="https://your-provider.example/v1"
export LLM_API_KEY="your-key"
export LLM_MODEL="provider/model-for-code-agent"
export ROUTER_MODEL="provider/model-for-routing" # optional
export EVAL_MODEL="provider/model-for-evaluation" # optional
```

(совместимость со smolagents не гарантирована)
## Запуск
(проще всего с `HF_TOKEN`, 100% совместимо)
```bash
python agent.py
```

- `/clear` clears the current conversation.
- `/exit` or `/quit` ends the chat.

## LLM-судья

По умолчанию evaluation сравнивает профили `good` и `bad` на одних и тех же
кейcах:

```bash
python evaluation.py
```

Текущий профиль `good` использует активные инструкции агента, роутера и tools.
Профиль `bad` — намеренно упрощённый набор для эксперимента. Заполни
`bad_profile` в `agent_prompts.yaml` и `BAD_TOOL_DESCRIPTIONS` в `config.py`
слабыми вариантами инструкций. Пока эти поля не заполнены, evaluation остановится
до API-вызовов и укажет недостающие части. Для проверки только хорошего профиля:

```bash
python evaluation.py --profiles good --limit 2
```

Профили запускаются попарно на каждом кейсе. В JSON-отчёте результаты сгруппированы
по метке `prompt_profile`; судья использует один и тот же системный промпт для
обоих вариантов.

Параметры:

```text
--cases PATH   JSONL case file (default: eval_cases.jsonl)
--output PATH  JSON report (default: evaluation_results.json)
--limit N      Run only the first N cases
--model NAME   Evaluator model (default: EVAL_MODEL or the configured default)
--profiles     Comma-separated profiles, e.g. good,bad
```
Запуск одного кейса на одном профиле - минимум три обращения к модели, полный запуск evaluation - 60. 

## Проблемы

- переписывание промта доверено рутеру. смысл промта может исказиться
- рутер может менять температуру агента (сделано для эксперимента, но не подходит для реальной работы)
- если ретрив не нашел релевантных результатов, он вернет `None` но модель может попытаться сымитировать rag
  instruction. ищу способ исправить
- `CodeAgent` не является полноценной os-песочницей
- вывод агента (из-за настроек verbose) может содержать нежелательную информацию
- иногда агент начинает кодить сам себе инструменты (даже на низкой температуре и на разных моделях), но хотелось бы детерменированности. ищу способ исправить
- требуется еще один этап рефакторинга для соблюдения единого оформления (не в каждом файле инфострока сверху, не в каждом файле регионы и тп)
- хотелось бы уменьшить цену запросов, но slm плохо справляются с задачей рутинга
