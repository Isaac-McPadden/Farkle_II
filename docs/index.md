# Documentation

Start with the [README](../README.md) for installation and workflow ownership.

| Reference | Purpose |
| --- | --- |
| [CLI](cli.md) | Commands, overrides, ownership and H2H approval |
| [Rules](rules.md) | Scoring, turns and official winner semantics |
| [Configuration](config_reference.md) | Public fields and paths |
| [Configuration option audit](config_option_audit.md) | Fixed protocol fields, unused options, and real controls |
| [Statistical methods](statistical_methods.md) | Conditioning, uncertainty and permitted claims |
| [Artifacts](data_artifacts.md) | Scopes, authenticated publication and recovery |
| [RNG](rng_contract.md) | Coordinate ownership and diagnostic interpretation |
| [Turn and row contract](turn_and_row_contract.md) | Observational units and outcomes |
| [Resource controls](os_memory_enforcement.md) | OS boundaries and failure handling |
| [Operations](operations.md) | Qualified launch/resume and capacity limits |
| [Terminology](terminology.md) | Maintained product language |
| [Contributor orientation](codex_context/context_prompt.md) | Source ownership, invariants and tests |
| [Development history](development_history.md) | Decisions and retained acceptance evidence |
| [Lifetime game estimate](game_count.md) | Roughly 1.7 billion games and counting basis |

`implementation/` retains frozen specifications, runtime-read recipes and
final first-release evidence. `remediation/` retains historical machine-readable
measurements. Their task numbers belong to different sequences; use the history
instead of treating frozen specifications as a current task queue.
