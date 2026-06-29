Add configurable per-agent iteration limits to prevent runaway tool-calling
loops, OOM kills, and unbounded API costs. This addresses a gap where the
existing heuristic loop detection (content-hash dedup) fails when volatile
messageIds defeat all detection paths.

The implementation adds:

- IterationBudget class (src/agents/iteration-budget.ts) with consume/refund
  semantics and configurable limits (default: 90 for parent agents, 50 for
  subagents)
- onBeforeToolCallingRound hook in agent-core's agent loop, called before
  each tool-calling round with a 1-based counter
- Budget integration in the embedded agent runner: creates a budget from
  resolved config, wires the hook to consume iterations, and produces a
  structured budget_exhausted result when exhausted
- Optional forced-summary grace call: when enabled, injects a final LLM
  turn with a summary instruction before stopping, so the user gets a
  coherent status instead of an abrupt halt
- Full configuration support via openclaw.json:
  agents.defaults.iterationBudget (global) and per-agent overrides with
  enabled, maxIterations, subagentMaxIterations, and
  forceSummaryOnExhaustion fields
- Zod schema validation with .strict() for all new config fields
- Resolved agent config merging (shallow merge matching runRetries pattern)

Test coverage (193 tests across 5 files):
- iteration-budget.test.ts: 22 tests covering consume, refund, exhaustion,
  edge cases, and config resolution
- agent-scope.test.ts: 100 tests covering config merging and defaults
- budget-exhaustion.test.ts: 10 tests covering result builder and summary
  instruction
- zod-schema.agent-defaults.test.ts: 41 tests covering schema validation
- agent-loop.test.ts: 20 tests covering onBeforeToolCallingRound hook

Addresses: #6890, #75468, #89090, #79301, #75355, #77625, #74859, #82518

## What Problem This Solves

OpenClaw has no explicit iteration budget for agent runs. The agent loop runs until the LLM stops calling tools or a timeout fires. Existing loop detection is heuristic (content-hash dedup) which fails when volatile messageIds defeat all detection paths. This causes runaway tool-calling loops, OOM kills, and unbounded API costs.

Addresses: #6890, #75468, #89090, #79301, #75355, #77625, #74859, #82518

## Evidence

### Test Results


```
SUCCESS

Feature: Iteration Budget & Agent Loop Safety
Runtime Iteration: V1
Date: 2026-06-28

Build: PASS
Ported Feature Tests: 193/193 PASS
Relevant Shard Tests: 1,814+ PASS (2 full shards clean)

All ~59 test failures across 8 shards are pre-existing OpenClaw issues
(provider catalog compat, plugin mock gaps, timing flakes, SDK budget drift)
with zero overlap to the 20 files in our patch.

Patch: code_gen_runtime_V1.patch
Files: 20 (16 modified, 4 new)
Lines: +1,020 / -1

No code fixes were required. The ported code from code_gen_review_V3.patch
is correct as-is.
```


### Behavior Proof

<details>
<summary>Terminal output showing iteration budget stopping a runaway agent loop</summary>

```
================================================================================
BEHAVIOR PROOF — Iteration Budget Agent Loop Safety
================================================================================
PR:         openclaw/openclaw#97485
Branch:     feat/iteration-budget-agent-loop-safety
Generated:  2026-06-29
Round:      2
Script:     test/proof/iteration-budget-proof.ts
Command:    npx tsx test/proof/iteration-budget-proof.ts
================================================================================

Terminal output (stdout):

========================================================================
BEHAVIOR PROOF: Iteration Budget Stops Runaway Agent Loop
========================================================================

--- Scenario 1: Budget enforcement ---

Config resolved: {
  "enabled": true,
  "maxIterations": 5,
  "subagentMaxIterations": 50,
  "forceSummaryOnExhaustion": true
}

Budget created: maxTotal=5, remaining=5

Simulating agent loop (would run 100+ rounds without budget)...

  Round 1: allowed (used=1/5, remaining=4)
  Round 2: allowed (used=2/5, remaining=3)
  Round 3: allowed (used=3/5, remaining=2)
  Round 4: allowed (used=4/5, remaining=1)
  Round 5: allowed (used=5/5, remaining=0)
  Round 6: BLOCKED by budget (used=5/5, remaining=0)

RESULT: Budget stopped the loop after 5 iterations (configured max: 5)

--- Scenario 2: Fail-closed summary path (no refund) ---

Budget state: used=5, remaining=0
Summary instruction injected. Model responds...

  Summary tool attempt: BLOCKED (correct)
  Budget state: used=5, remaining=0

--- Scenario 3: Anti-pattern — refund before summary ---

Budget exhausted: used=3, remaining=0
(Anti-pattern) Refunding 1 iteration before summary...
After refund: used=2, remaining=1
  Tool attempt after refund: ALLOWED (this is the bug)

The refund anti-pattern allows tool execution after exhaustion.
The fix (no refund) prevents this by keeping the budget at 0.

--- Scenario 4: Subagent cap selection ---

Subagent run: cap=50 (from subagentMaxIterations), maxTotal=50
Parent run:   cap=90 (from maxIterations), maxTotal=90

========================================================================
ALL SCENARIOS PASSED — budget enforces hard iteration caps
========================================================================
```

</details>


### Test Coverage

Test files covering this feature:

- `src/agents/iteration-budget.test.ts` — core budget consume/refund/exhaustion
- `src/agents/agent-scope.test.ts` — config merging and defaults
- `src/agents/embedded-agent-runner/run/budget-exhaustion.test.ts` — result builder and summary instruction
- `src/config/zod-schema.agent-defaults.test.ts` — schema validation
- `packages/agent-core/src/agent-loop.test.ts` — onBeforeToolCallingRound hook
