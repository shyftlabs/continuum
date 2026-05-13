from dataclasses import dataclass, field


@dataclass
class DemoConfig:
    model: str = "gemini/gemini-2.5-flash"
    max_turns: int = 10
    enable_memory: bool = True
    enable_session: bool = True

    mode_descriptions: dict = field(default_factory=lambda: {
        "sequential": "question → research → write → edit  (pipeline)",
        "parallel":   "research topic from multiple angles simultaneously",
        "loop":       "keep refining until the answer says DONE",
        "scatter":    "split topic into N subtopics, analyse in parallel",
        "supervised": "write essay — supervisor retries if quality < 0.7",
        "planner":    "dynamic plan: LLM decides which steps to run",
        "debate":     "pro vs con on any topic — judge gives final verdict",
        "reflection": "write content — self-critique until PASS",
        "router":     "triage: route to researcher, writer, or fact-checker",
        "handoff":    "orchestrator breaks down task, hands off to researcher",
    })


default_config = DemoConfig()
