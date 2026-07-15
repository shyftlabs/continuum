| # | Scenario | Section | Without (tok) | With (tok) | Saved | Transforms |
|---|----------|---------|--------------:|-----------:|------:|------------|
| 1 | DB rows (failed orders) | efficiency | 5,440 | 4,409 | 19.0% | `router:protected:system_message, router:smart_crusher:0.00` |
| 2 | Logs — checkout-api | efficiency | 31,406 | 310 | 99.0% | `router:protected:system_message, router:search:0.01` |
| 3 | Search / RAG runbooks | efficiency | 10,086 | 3,571 | 64.6% | `router:protected:system_message, router:mixed:0.37` |
| 7 | Streaming path (logs) | efficiency | 31,329 | 314 | 99.0% | `router:protected:system_message, router:search:0.01` |
| 8 | Multi-tool (two log dumps) | efficiency | 62,682 | 571 | 99.1% | `router:protected:system_message, router:search:0.01, router:` |
| 11 | File read→write (stale, CCR ticket) | efficiency | 346 | 168 | 51.4% | `read_lifecycle:stale:service.yaml, router:excluded:tool, rou` |
| 4 | read tool (excluded) | safety | 301 | 301 | 0.0% | `router:protected:system_message, router:excluded:tool` |
| 10 | RAG context (system msg) | safety | 10,055 | 10,055 | 0.0% | `router:protected:system_message, router:protected:system_mes` |
| 9 | Postmortem prose (Kompress) | efficiency | 7,769 | 5,991 | 22.9% | `router:protected:system_message, router:text:0.78` |
| — | **Aggregate (efficiency)** | | **149,058** | **15,334** | **89.7%** | |
| 5 | Fail-open (dead sidecar) | resilience | — | — | — | sidecar down → compress errors, Continuum fail-opens (forwards uncompressed, run survives) |
| 6 | Anti-forgery | security | — | — | — | forged hash rejected |
