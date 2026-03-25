# Continuum vs. Competitors — Head-to-Head Comparison

## Feature Comparison Matrix

| Feature | Continuum | LangGraph | CrewAI | OpenAI SDK | Google ADK | AutoGen/AG2 | Pydantic AI | Semantic Kernel | Bedrock Agents |
|---|---|---|---|---|---|---|---|---|---|
| **Language** | Python | Python, JS | Python | Python, TS | Python, TS | Python | Python | C#, Python, Java | Python, SDKs |
| **Multi-LLM Support** | 100+ (LiteLLM) | Multi-model | Multi-model | OpenAI only | Gemini-first | Multi-model | Multi-model | Multi-model | Bedrock models |
| **Tool Protocol** | MCP native | Custom + MCP | MCP native | Function calling | Built-in | Function calling | MCP native | MCP + Plugins | Lambda/API |
| **Multi-Agent** | Handoffs + Workflows | Graph nodes | Role-based | Handoffs | Hierarchical | Conversational | Composition | Roadmap | Managed |
| **Memory (Long-term)** | mem0 + Qdrant (4 scopes) | Checkpointers | Internal | Sessions | Persistent state | MemoryStream | DI-based | Built-in layer | Built-in |
| **Memory (Short-term)** | Redis sessions | Graph state | Internal | Conversation | Event loop | Chat history | State | Chat history | Built-in |
| **Durable Workflows** | Temporal | Checkpoints | No | No | No | No | No | No | Managed |
| **Observability** | Langfuse (built-in) | LangSmith | Basic | Built-in tracing | Cloud Logging | Basic | Basic | Azure Monitor | CloudWatch |
| **Structured Output** | Pydantic + JSON Schema | Custom | Basic | Guardrails | Basic | Basic | Pydantic (core) | Basic | Basic |
| **Context Compression** | Automatic | Manual | No | No | No | No | No | No | Managed |
| **Lifecycle Hooks** | 5 hooks | Callbacks | No | Guardrails | No | No | Hooks | Events | No |
| **Prompt Engineering** | Template vars, examples, modifiers | Manual | Role prompts | Instructions | Instructions | System messages | Instructions | Prompts | Instructions |
| **Human-in-Loop** | Temporal approval gates | Interrupt nodes | No | No | No | Human proxy | No | No | No |
| **Health Checks** | Built-in | No | No | No | No | No | No | No | Managed |
| **DI Container** | Built-in | No | No | No | No | No | Built-in | Built-in | No |
| **Self-Hosted** | Yes (Docker Compose) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No (AWS only) |
| **Community Size** | Internal/Private | ~97k stars | ~46k stars | Official OpenAI | Google-backed | ~55k stars | ~15k stars | ~27k stars | Enterprise |

---

## Where Continuum Wins

### 1. Memory Architecture
Continuum's 4-scope memory isolation (SHARED, USER, AGENT, RUN) is more sophisticated than any competitor. LangGraph has checkpointers but no built-in semantic memory. CrewAI and OpenAI SDK have basic session management. Continuum combines short-term (Redis) + long-term (mem0/Qdrant) with automatic fact extraction and semantic retrieval.

### 2. Durable Workflows via Temporal
No other open-source agent framework integrates Temporal for crash-proof, long-running orchestration. LangGraph has checkpoints (which recover state), but Temporal provides true durable execution — workflows survive process crashes, network failures, and deployments. This is a significant enterprise differentiator.

### 3. MCP-Native from Day One
While LangChain, AutoGen, and OpenAI SDK still use custom tool interfaces (with MCP as an add-on), Continuum was built MCP-native. Any MCP server works out of the box — no adapters, no wrappers. This future-proofs the tool ecosystem as MCP becomes the industry standard.

### 4. Production Infrastructure Bundle
Continuum ships with Docker Compose for the entire stack: Langfuse (observability), Qdrant (vectors), Redis (sessions), PostgreSQL, ClickHouse, Temporal. No other framework provides this level of "batteries included" infrastructure for production deployments.

### 5. Automatic Model Compatibility
The LLM client transparently handles provider-specific quirks (e.g., auto-disabling JSON mode when tools are present on Gemini). Other frameworks leave this to the developer.

### 6. Prompt Engineering as First-Class
Template variables, few-shot examples, and instruction modifiers are built into `BaseAgent`. Most frameworks treat prompting as "just write a string." Continuum makes dynamic, context-aware prompt construction a core feature.

---

## Where Competitors Win

### LangGraph over Continuum
- **Community & ecosystem**: 97k stars, massive plugin ecosystem, LangSmith for managed observability
- **Graph-based orchestration**: More expressive for complex, non-linear workflows than Continuum's declarative patterns
- **Production track record**: Proven at Klarna, Cisco, and other large enterprises
- **JavaScript support**: Continuum is Python-only

### CrewAI over Continuum
- **Simplicity**: Role-based metaphor is more intuitive for non-experts
- **Scale proof**: 12M+ daily executions in production
- **A2A protocol**: Inter-framework agent communication support
- **Lower learning curve**: Faster time-to-first-agent

### OpenAI Agents SDK over Continuum
- **Minimalism**: Radically simpler API surface
- **OpenAI model optimization**: Best experience with GPT models
- **Built-in guardrails**: Safety layer included
- **TypeScript support**: First-class JavaScript/TypeScript option

### Pydantic AI over Continuum
- **Type safety**: Deeper IDE integration, write-time error detection
- **Testing story**: Dependency injection designed for testability
- **Cleaner API**: Less configuration surface area

### Amazon Bedrock over Continuum
- **Zero ops**: Fully managed, no infrastructure to maintain
- **Policy controls**: Natural language access policies via Cedar
- **AWS integration**: Native connection to the AWS ecosystem

---

## Positioning Map

```
                    High Control / Code-First
                           │
              LangGraph ●  │  ● Continuum
                           │
         Pydantic AI ●     │     ● Haystack
                           │
    Simple ─────────────────┼──────────────── Complex
                           │
           CrewAI ●        │        ● AutoGen/AG2
                           │
        OpenAI SDK ●       │     ● Semantic Kernel
                           │
                    Low Control / Managed
                           │
               Dify ●      │      ● Bedrock Agents
```

---

## Continuum's Competitive Moat

Continuum occupies a unique position: **production-grade, self-hosted, multi-LLM agent framework with enterprise infrastructure built in**. Its closest competitor is LangGraph, but Continuum differentiates on:

1. **Temporal integration** — no other framework has durable workflow execution
2. **Memory isolation scopes** — purpose-built for multi-tenant SaaS
3. **Infrastructure bundle** — Docker Compose with observability, vector DB, sessions, durable workflows
4. **MCP-native tools** — future-proof tool ecosystem without adapter layers
5. **Model-agnostic with compatibility management** — works with 100+ providers and handles their quirks

The trade-off is community size and ecosystem breadth. LangChain/LangGraph has 97k stars and hundreds of integrations. Continuum is a focused, opinionated SDK that prioritizes production readiness over ecosystem breadth.

---

## Recommendation for Positioning

Continuum should position itself as: **"The production-grade agent SDK for teams that need enterprise memory, durable workflows, and multi-LLM flexibility — without managed cloud lock-in."**

Key differentiators to emphasize in messaging:
- "Temporal-powered durable workflows" (unique in the market)
- "4-scope memory isolation for multi-tenant systems" (unique depth)
- "MCP-native tool ecosystem" (future-proof)
- "100+ LLM providers, zero lock-in" (via LiteLLM)
- "Full observability stack included" (Langfuse, not a paid add-on)
