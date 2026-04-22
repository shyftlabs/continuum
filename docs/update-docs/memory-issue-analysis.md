# Memory mechanism and issues analysis

## 1. Memory Control mechanism

The SDK exposes **two memory layers** that are independently controllable per-agent:


| Layer                 | Backend                             | Purpose                                                                    | Persistence                                        |
| --------------------- | ----------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------- |
| **Long-term memory**  | mem0 + vector store (Qdrant/Milvus) | Semantic facts extracted from conversations (e.g. "User works in finance") | Permanent (until explicitly deleted)               |
| **Short-term memory** | Redis session store                 | Recent conversation turns (user + assistant messages)                      | TTL-based (default 7 days), scoped to `session_id` |


### Prompt Assembly Order

`MessageBuilder.prepare_messages()` assembles the final LLM prompt in this order:

```
1. System prompt
2. ReAct scaffold (if enabled)
3. Tool context (if exists)
4. ★ Long-term memories (mem0)          ← AgentMemoryConfig.search_memories
5. Pipeline context (only for multi-agent workflows)
6. ★ Short-term history (Redis)         ← AgentConfig.session_history_turns
7. RAG context (if provided)
8. User input (current query)
```

### Controlling Long-term Memory (mem0)

Configured per-agent via `AgentMemoryConfig`:

```python
from orchestrator.agent.config import AgentMemoryConfig, AgentConfig
from orchestrator.agent.types import MemoryScope

agent_config = AgentConfig(
    memory=AgentMemoryConfig(
        # === READ (loading memories into LLM context) ===
        search_memories=True,          # False = skip long-term memory entirely
        search_limit=5,                # How many memories to inject (default: 5)
        search_scope=MemoryScope.USER, # USER | AGENT | CONVERSATION | SHARED
        search_threshold=0.0,          # Minimum similarity score

        # === WRITE (storing new facts after each turn) ===
        store_memories=True,           # False = don't extract/store facts
        store_scope=MemoryScope.USER,
        extraction_prompt=None,        # Custom fact extraction prompt for mem0
    ),
)
```

**Global kill switch** in `.env`:

```env
MEMORY_ENABLED=false          # Disables the entire mem0 system
MEMORY_ISOLATION=user         # "shared" | "user" | "agent" | "conversation"
MEMORY_SEARCH_LIMIT=5         # Default number of memories to retrieve
```

### Controlling Short-term Memory (Redis Session History)

Configured per-agent via `AgentConfig`:

```python
agent_config = AgentConfig(
    session_history_turns=20,   # Number of turns to load (default: 20)
                                 # Set to 0 to disable short-term memory loading
    log_to_session=True,         # Whether to write this agent's output to Redis
)
```

**Global settings** in `.env`:

```env
SESSION_ENABLED=false           # Disables the entire Redis session system
SESSION_TTL_SECONDS=604800      # 7 days (default)
SESSION_MAX_MESSAGES=1000       # Max messages per session
```

### Quick Examples

**Read-only agent** (loads memories but never writes new facts):

```python
AgentMemoryConfig(search_memories=True, store_memories=False)
```

**Fully stateless agent** (no memory of any kind):

```python
AgentConfig(
    memory=AgentMemoryConfig(search_memories=False, store_memories=False),
    session_history_turns=0,
    log_to_session=False,
)
```

**Minimal context window usage** (useful for cheap/small models):

```python
AgentConfig(
    memory=AgentMemoryConfig(search_memories=True, search_limit=2),
    session_history_turns=3,  # Only last 3 turns
)
```

**Conversation-scoped memory** (memories isolated per chat window):

```python
AgentConfig(
    memory=AgentMemoryConfig(
        search_memories=True,
        search_scope=MemoryScope.CONVERSATION,
        store_scope=MemoryScope.CONVERSATION,
    ),
)
# At runtime, pass conversation_id:
response = await runner.run(agent, user_input, conversation_id="chat-window-abc123")
```

**Agent-scoped memory** (each agent has its own memory silo):

```python
AgentConfig(
    memory=AgentMemoryConfig(
        search_scope=MemoryScope.AGENT,
        store_scope=MemoryScope.AGENT,
    ),
)
```

---

## Issue 1: no `conversation_id` from frontend

> [!IMPORTANT]
> When a product integrates Continuum, the **frontend must pass `conversation_id`** on every request. This is the single most important integration contract.

### What is `conversation_id`?

`conversation_id` is a **stable identifier for a chat window**. It maps 1:1 to what the user sees as a conversation in the UI:

```
┌─────────────────────────────────────────────┐
│  Chat Window (frontend)                     │
│  conversation_id = "conv-a1b2c3d4"          │
│                                             │
│  User: "What tax deductions can I claim?"   │  ← request 1 (run_id = "run-001")
│  Bot:  "Here are the common deductions..."  │
│                                             │
│  User: "What about home office?"            │  ← request 2 (run_id = "run-002")
│  Bot:  "Since you're a freelancer..."       │    (remembers user context from request 1)
│                                             │
│  User: "Thanks, what about vehicle costs?"  │  ← request 3 (run_id = "run-003")
│  Bot:  "Based on your freelance status..."  │    (still has full conversation context)
└─────────────────────────────────────────────┘
```

- `**run_id**` is regenerated on every request — it's ephemeral
- `**conversation_id**` persists across all requests in the same chat window — it's stable
- `**session_id**` is for Redis short-term history (may or may not equal `conversation_id`)

### Why it matters

Without `conversation_id`, the SDK cannot:

1. **Scope long-term memories to a conversation** — memories from unrelated chat windows would leak across
2. **Load the correct session history** — the agent wouldn't know which past messages belong to this conversation
3. **Support conversation-level isolation** — `MEMORY_ISOLATION=conversation` requires this ID

### How to pass it

**Backend API endpoint** (the product's router/controller):

```python
@app.post("/api/chat")
async def chat(request: ChatRequest):
    response = await runner.run(
        agent=my_agent,
        input=request.message,
        session_id=request.session_id,            # Redis session key
        conversation_id=request.conversation_id,  # ← REQUIRED from frontend
        user_id=request.user_id,
    )
    return {"response": response.content}
```

**Frontend** (React/Next.js example):

```typescript
// Generate conversation_id when user opens a new chat window
const conversationId = crypto.randomUUID();

// Send on every message in that chat window
const response = await fetch("/api/chat", {
  method: "POST",
  body: JSON.stringify({
    message: userInput,
    conversation_id: conversationId,  // stable for entire chat window lifetime
    session_id: sessionId,            // can be same as conversation_id, or separate
    user_id: currentUser.id,
  }),
});
```

### Lifecycle


| Event                               | Action                                                                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------------------- |
| User opens new chat window          | Frontend generates new `conversation_id` (UUID)                                                 |
| User sends message in existing chat | Frontend sends the **same** `conversation_id`                                                   |
| User opens a different chat window  | Frontend generates a **different** `conversation_id`                                            |
| User closes and reopens same chat   | Frontend should **restore** the original `conversation_id` (persist in localStorage or backend) |


### What happens if `conversation_id` is not passed

> [!CAUTION]
> Not passing `conversation_id` causes **memory leakage across chat windows** in both memory layers. This is the most common integration mistake.

#### Long-term memory (mem0): facts leak across conversations

The SDK logs a warning and falls back to **unscoped** behavior:

```
WARNING: memory_isolation='conversation' but context.conversation_id is None —
memory search will be unscoped. Pass conversation_id when calling runner.run().
```

- Facts extracted in Chat Window A are visible in Chat Window B
- The agent may reference information from an unrelated conversation, confusing the user
- `delete_all` for a conversation cannot target just that conversation's facts

#### Short-term memory (Redis session): chat history shared across windows

The Redis session provider uses `conversation_id` to derive the Redis key via `_compute_session_id()` in `session/providers/redis.py`:

```python
def _compute_session_id(self, session_id, user_id, conversation_id) -> str:
    if session_id:                        # explicit → use as-is
        return session_id
    if conversation_id and user_id:       # ← "c:{conversation_id}:u:{user_id}"
        return f"c:{conversation_id}:u:{user_id}"
    if user_id:                           # user only → "u:{user_id}"
        return f"u:{user_id}"
    return generate_session_id()          # fallback → random UUID
```

**With `conversation_id`:** each chat window gets its own Redis key → isolated history ✅

```
Chat Window A (conversation_id="conv-aaa", user_id="user-123")
  → Redis key: "c:conv-aaa:u:user-123"

Chat Window B (conversation_id="conv-bbb", user_id="user-123")
  → Redis key: "c:conv-bbb:u:user-123"
```

**Without `conversation_id`:** falls back to `"u:{user_id}"` → all chat windows share one session ❌

```
Chat Window A (user_id="user-123")  ──┐
                                      ├── Redis key: "u:user-123" → SHARED history
Chat Window B (user_id="user-123")  ──┘

User opens Chat Window A → asks about taxes
User opens Chat Window B → asks about recipes

Result: Chat Window B sees "What tax deductions can I claim?" in its context.
The agent gives a confused response mixing tax and recipe topics.
```

#### Summary of dangers


| Layer                     | Without `conversation_id`                                        | Risk                                                                    |
| ------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Short-term (Redis)**    | All chat windows for the same user share Redis key `u:{user_id}` | Agent sees messages from other chat windows, causing confused responses |
| **Long-term (mem0)**      | Facts stored/searched globally for the user                      | Agent recalls facts from unrelated conversations                        |
| **Privacy**               | Sensitive data from one conversation leaks into another          | User sees information they shared in a private context                  |
| **Multi-agent workflows** | Pipeline context from one conversation bleeds into another       | Workflow state becomes unpredictable                                    |


## Issue 2: Poor setting or orchestration in multi-agent mode

> Solution: understand your requirements on specific projects first, then based on them, choose whehter to disable load long-term memory or short-term memory (session in redis) for your specific agents; you can also choose what facts should be saved to your long-term memory

### Observed Behavior: Bad Case in Sequential Workflow Mode

In a **sequential pipeline** (pet shop example using `SequentialAgent`), the user sends a simple greeting `"hi"`, but the pipeline acts as if the user asked to buy dog food:

```
[system] You are a friendly pet shop assistant. Read the prior pipeline steps from context
         and write a single, clear summary for the user: what was found, what was recommended,
         and what was done. Keep it less than 3-4 sentences.
[system] User profile (long-term preferences and context):
- Wants to buy dog food
- Looking for dog toys
- Favourite color is blue

[user] Original request: hi

Step 1 (search-agent): Hello! How can I help you today?

Step 2 (recommend-agent): I'm looking for some dog food.

Step 3 (cart-agent): We have "Dog Food (Dry) 5kg" for $29.99. Would you like to add it to your cart?
```

### Root Cause

The issue is that **long-term memory facts are injected into every sub-agent's prompt**, and sub-agents treat stored preferences as the user's **current intent**.

Here's how the prompt is assembled for each sub-agent in the pipeline (from `MessageBuilder.prepare_messages()`):

```
1. System prompt         → "You are a friendly pet shop assistant..."
2. ★ Long-term memories  → "Wants to buy dog food", "Looking for dog toys", "Favourite color is blue"
3. Pipeline context      → Prior steps' outputs (grows with each step)
4. User input            → "hi" (only in step 1; subsequent steps get previous agent's output)
```

The problem path:

```
Step 1 (search-agent):
  Prompt = system + memories["Wants to buy dog food"] + user["hi"]
  → Agent sees "hi" but also sees "Wants to buy dog food" in memory
  → Responds: "Hello! How can I help you today?" (benign — just greets)

Step 2 (recommend-agent):
  Prompt = system + memories["Wants to buy dog food"] + pipeline_context[step 1] + input["Hello! How can I help you today?"]
  → Agent sees memory fact "Wants to buy dog food" and interprets it as current intent
  → Responds: "I'm looking for some dog food." (WRONG — fabricates user intent from memory)

Step 3 (cart-agent):
  Prompt = system + memories["Wants to buy dog food"] + pipeline_context[steps 1-2] + input["I'm looking for some dog food."]
  → Now the pipeline has snowballed: the fabricated intent is treated as real
  → Responds: 'We have "Dog Food (Dry) 5kg" for $29.99...' (adds product based on hallucinated intent)
```

### Why This Happens
 **Memory facts look like instructions.** They appear as a system message ("User profile...") above the user input. LLMs interpret system messages as authoritative, so "Wants to buy dog food" gets treated as the user's current need.

### Potential Solutions

#### Option A: Disable long-term memory for intermediate sub-agents (Recommended)

Only the first agent in the pipeline should load long-term memory. Intermediate agents should rely on pipeline context:

```python
# Product-level fix: configure sub-agents with search_memories=False
search_agent = BaseAgent(
    name="search-agent",
    memory=AgentMemoryConfig(search_memories=True),   # ← loads memory
)
recommend_agent = BaseAgent(
    name="recommend-agent",
    memory=AgentMemoryConfig(search_memories=False),  # ← no memory
)
cart_agent = BaseAgent(
    name="cart-agent",
    memory=AgentMemoryConfig(search_memories=False),  # ← no memory
)
```

#### Option B: SDK-level fix — workflow automatically disables memory for non-first agents

The `SequentialAgent.execute()` could temporarily disable `search_memories` for agents after step 1, similar to how it already disables `log_to_session`:

```python
# In sequential.py, analogous to the log_to_session pattern:
_orig_mem = {a.name: a.memory_config.search_memories for a in self.agents}
for i, a in enumerate(self.agents):
    if i > 0:  # only first agent loads memory
        a.memory_config.search_memories = False
```

#### Option C: Controlling What Gets Saved to Long-Term Memory (mem0)

Continuum supports **3 levels of control** over what facts get saved to long-term memory, which can help prevent memory contamination or sensitive data leakage in the first place:

##### 1. `extraction_prompt` — Tell mem0 **what to extract**
Controls the LLM prompt that mem0 uses internally to decide what facts are worth extracting from a message.

```python
AgentMemoryConfig(
    extraction_prompt=(
        "Extract ONLY product preferences and purchase history. "
        "Ignore greetings, small talk, and generic questions."
    ),
)
```
**When it runs:** Before any facts are stored — mem0 uses this prompt to decide what to extract from the conversation.

##### 2. `pre_store_filter` — **Delete facts that shouldn't be kept** (post-extraction)
A callback that receives the list of extracted fact texts and returns only the ones allowed to stay. Facts not returned are **deleted from the vector store**.

```python
def pii_filter(facts: list[str]) -> list[str]:
    """Block facts containing PII."""
    blocked_patterns = ["SSN", "social security", "credit card", "password"]
    return [
        f for f in facts
        if not any(p.lower() in f.lower() for p in blocked_patterns)
    ]

AgentMemoryConfig(
    pre_store_filter=pii_filter,
)
```
**When it runs:** After mem0 extracts and stores facts → filter checks them → non-allowed facts are deleted via `memory_client.delete(fact_id)`. *(Note: this is a store-then-delete pattern because mem0 doesn't expose a pre-storage hook).*

##### 3. `on_stored` — **React after facts are stored**
A callback fired with the final list of stored facts (after filtering). Use for logging, analytics, or triggering side effects.

```python
def log_stored(facts: list[str]):
    print(f"Stored {len(facts)} facts: {facts}")

AgentMemoryConfig(
    on_stored=log_stored,
)
```

##### Also: Message type filtering
You can also control **which message roles** trigger fact extraction:

```python
AgentMemoryConfig(
    store_user_messages=True,       # extract facts from user messages
    store_assistant_messages=False, # skip assistant responses (no facts extracted)
)
```
