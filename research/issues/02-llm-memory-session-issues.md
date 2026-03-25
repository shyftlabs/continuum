# LLM Client, Memory & Session Layer Issues

---

## CRITICAL

### 1. Silent Token Counting Returns 0 on Any Exception
**File:** `src/orchestrator/llm/client.py` (~lines 999-1003)

`count_tokens()` returns `0` on ANY exception (network errors, model not found, invalid format). This causes the context compression logic to believe messages have 0 tokens, potentially discarding or compressing all valid messages. The fallback estimate in `context_window.py` (line 263) is never reached because the client never raises.

**Impact:** Messages silently discarded. Complete loss of conversation context in production.

**Fix:** Raise on failure, or return a conservative estimate instead of 0.

---

### 2. Missing JSON Parsing Error Handling in Redis Provider
**File:** `src/orchestrator/session/providers/redis.py` (~lines 231, 283, 389, 510, 521, 575)

Multiple `json.loads()` calls on Redis values have NO try/catch. If Redis data is corrupted (encoding issues, partial writes, manual tampering), a `JSONDecodeError` crashes the entire session operation with no recovery path.

**Impact:** Single corrupted Redis record breaks ALL session operations for that user. Potential DoS vector.

**Fix:** Wrap all `json.loads()` in try/catch. Log and skip corrupted entries.

---

### 3. Streaming Generator Not Properly Cleaned Up on Error
**File:** `src/orchestrator/llm/client.py` (~lines 599-610, 915-930)

Both `chat_stream_sync()` and `chat_stream()` use generators. If an exception occurs mid-iteration, the generator is left open — the response stream generator goes unclosed, causing HTTP connection leaks.

**Impact:** Long-running processes accumulate unclosed connections. Connection pool exhaustion in production.

**Fix:** Use try/finally in generators or context managers for cleanup.

---

### 4. Race Condition in LLMClient Rate Limiter
**File:** `src/orchestrator/llm/client.py` (~lines 57-77)

TOCTOU race in `_LLMRateLimiter.acquire()`. Between checking `self.tokens < 1` and sleeping, another coroutine could decrement tokens. Multiple coroutines can pass the check simultaneously, making rate limiting ineffective under concurrent load.

**Impact:** Rate limiting fails. Request bursts exceed configured RPM.

---

## HIGH

### 5. Truncation Strategy May Lose System Prompt
**File:** `src/orchestrator/llm/context_window.py` (~lines 309-336, 394-440)

`_truncate_smart()` attempts to find the first user message but doesn't handle the edge case where NO user message exists (only system and assistant messages). Result list is incorrectly built, potentially duplicating messages or losing context.

**Impact:** System prompts and critical context can be truncated.

---

### 6. Exponential Memory Leak in Summary Cache
**File:** `src/orchestrator/llm/context_management.py` (~lines 130-168)

`SummaryCache._cache` uses message content for key generation with no size limit on the cache dict. Messages can contain arbitrarily large content. TTL removes expired entries but doesn't limit total cache size.

**Impact:** Memory exhaustion; potential DoS.

**Fix:** Add max_size parameter and LRU eviction policy.

---

### 7. Context Manager Global Initialization Race Condition
**File:** `src/orchestrator/llm/context_window.py` (~lines 520-534)

Double-checked locking pattern is flawed. After lock acquisition, there's no re-check, so the global could be initialized twice from two threads.

**Impact:** Multiple ContextWindowManager instances; inconsistent state.

---

### 8. Mem0 Provider Doesn't Implement Sync Methods
**File:** `src/orchestrator/memory/providers/mem0.py` (entire file)

`Mem0Provider` extends `BaseMemoryProvider` with abstract `add_sync()`, `search_sync()`, etc. but doesn't implement them. Synchronous memory operations fail with NotImplementedError or AttributeError.

**Impact:** Synchronous memory operations broken for all users.

---

### 9. Session Message Storage Race Condition
**File:** `src/orchestrator/session/providers/redis.py` (~lines 386-442)

Between checking session existence and incrementing message count, another coroutine could delete the session. If LTRIM (sliding window) succeeds but SETEX fails, message count becomes out of sync with actual Redis list length. No Redis MULTI/EXEC transaction wraps these operations.

**Impact:** Inconsistent session state; message counts don't match actual messages.

---

### 10. Memory Context Missing Initialization Check in SessionClient
**File:** `src/orchestrator/session/client.py` (~lines 117-130)

`memory_client` property doesn't check if the retrieved client is None before using it. If memory initialization fails, operations crash with AttributeError.

---

### 11. Missing Null Check for Streaming Chunks
**File:** `src/orchestrator/llm/types.py` (~lines 200-223)

`StreamChunk.from_litellm_chunk()` doesn't validate that `chunk.choices[0].delta.content` exists before accessing. None content propagates to callers.

---

### 12. Context Management Doesn't Handle Empty Message List
**File:** `src/orchestrator/llm/context_management.py` (~lines 259-301)

`_compress_summarize()` doesn't handle empty message lists or when `keep_recent_messages >= len(messages)`. Still attempts token counting on empty list.

---

## MEDIUM

### 13. Bare Exception Catching in LLMClient Initialization
**File:** `src/orchestrator/llm/client.py` (~lines 158, 238)

`except Exception:` at setup_langfuse silently swallows all errors including configuration issues.

---

### 14. Token Counting Fallback Estimate is Inaccurate
**File:** `src/orchestrator/llm/context_window.py` (~lines 262-266)

Character-based fallback (`total_chars // 4`) is crude. Highly inaccurate for non-English text, structured data, and code.

---

### 15. Missing Validation for Negative Max Tokens
**File:** `src/orchestrator/llm/context_window.py` (~lines 58-68)

`ModelLimits` doesn't validate `max_tokens > 0`. Can cause division by zero or negative effective limits.

---

### 16. No Timeout on asyncio.to_thread() Calls in mem0 Provider
**File:** `src/orchestrator/memory/providers/mem0.py` (~lines 242, 291, 312, 345)

All `asyncio.to_thread()` calls run indefinitely. If mem0 operations hang, the thread pool becomes exhausted.

---

### 17. LLM Client Auto-Session Loading Modifies Message Order
**File:** `src/orchestrator/llm/client.py` (~lines 709-730)

Auto-loading conversation history prepends old messages before new ones. If context compression then runs, it might remove the current turn while keeping stale history.

---

### 18. Memory Storage Errors Silently Logged But Not Reported
**File:** `src/orchestrator/session/client.py` (~lines 167-179)

Memory storage failures are logged but don't update observability metrics. Users won't know facts weren't stored.

---

### 19. JSON Encoding Assumes All Message Fields Are Serializable
**File:** `src/orchestrator/session/providers/redis.py` (~lines 434, 441)

`json.dumps(session_message.to_dict())` will fail if ChatMessage contains non-serializable objects (e.g., datetime not converted to strings). No error handling for serialization failures.

---

### 20. Rate Limiter Doesn't Handle RPM=0
**File:** `src/orchestrator/llm/client.py` (~lines 60-77)

If `rate_limit_rpm=0`, division by zero in `elapsed * (self.rpm / 60.0)`. Should validate RPM > 0 in LLMConfig.

---

### 21. Global Memory Client Double Initialization
**File:** `src/orchestrator/memory/client.py` (~lines 493-510)

Same flawed double-checked locking as context window manager. No re-check after lock acquisition.

---

### 22. Memory Client Provider Creation Swallows ImportError
**File:** `src/orchestrator/memory/client.py` (~lines 117-133)

ImportError is caught and logged, but provider stays uninitialized. `is_enabled` returns False with no clear indication why.

---

### 23. Session Cleanup Doesn't Wait for Async Operations
**File:** `src/orchestrator/session/providers/redis.py` (~lines 701-731)

`close()` calls `await pool.aclose()` while background tasks may still be using the connection. No graceful shutdown wait period.

---

### 24. Streaming Chunk Finish Reason Not Propagated
**File:** `src/orchestrator/llm/client.py` (~lines 851-930)

`chat_stream()` doesn't track finish_reason across chunks. Caller can't determine if stream ended normally or was interrupted until consuming all chunks.
