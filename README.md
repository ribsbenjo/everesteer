# everesteer
This repo is for the competition everesteer, we want to make an agentic trading bot, that beats our competition.

## What's in here

| Path | What it is |
|---|---|
| `Everesteer_backtest/` | Backtest harness: `main.py` (our forecast), `round.py` (one command per round), `backtest.py` (walk-forward replay, leak test, stats). See its README. `data/prices.csv` is synthetic placeholder data. |
| `.claude/skills/agentic-trading-research/` | Claude skill distilled from the five papers below: look-ahead leakage, execution-layer safety, adversarial robustness of multi-agent pipelines, evaluation and reporting standards, plus scripts (`leak_check.py`, `execution_gate.py`, `robust_stats.py`). Claude Code picks it up automatically when run from this folder. |
| `agentic-trading-research.skill` | The same skill packaged for installing in the Claude app. |
| Papers (not stored here) | [Agentic trading survey](https://arxiv.org/abs/2605.19337), [Look-ahead-freedom](https://arxiv.org/abs/2607.04958), [Survivability-Aware Execution](https://arxiv.org/abs/2603.10092), [Poisoning Agentic Alpha](https://arxiv.org/abs/2608.24069), [The Agentic ETF](https://arxiv.org/abs/2610.06856). |
| `Everesteer Night Playbook.html`, `everesteer-cheat-sheet.html` | Night-of playbook and cheat sheet. |

## Quick commands

```bash
cd Everesteer_backtest && python round.py
```

```bash
python .claude/skills/agentic-trading-research/scripts/leak_check.py lint Everesteer_backtest/main.py
```
