# Small agent

## Running the chat

Run `python agent.py` to start the command-line chat. Use `/clear` to clear the
current conversation and `/exit` (or `/quit`) to stop it.

## Critical: conversation history is per process and in memory only

The CLI stores visible user messages and final assistant answers in a local
`history` list for the lifetime of that process. That history is passed to the
router to resolve follow-up references and to the agent as conversational
context. Internal reasoning, generated code, and tool observations are not
copied into this conversation history.

The history is not written to disk, does not survive process restarts, and is
not shared between separate CLI runs. The agent and its model are created for
each turn; conversational continuity comes from explicitly passing this
visible history, not from reusing the agent's internal execution memory.

**Do not replace the per-session history with one module-level global list in a
multi-user server.** That could mix users' conversations and disclose one
user's messages to another. A web deployment must keep history in isolated
session state keyed by a verified session/user identity, with appropriate
access controls. Persisting history across restarts would additionally require
an explicit retention policy and secure storage.

## Current limitations

- The knowledge base and similarity cutoff are demonstration values. Evaluate
  retrieval quality before relying on it for factual answers.
- RAG searches using the router's rewritten current query. Conversation history
  helps resolve follow-ups, but retrieval quality still needs its own tests.
- `CodeAgent` executes model-generated Python. Its local executor should not be
  treated as an operating-system security sandbox for untrusted users or
  retrieved documents.
