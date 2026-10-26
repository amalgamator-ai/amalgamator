"""Parallel Research Iteration (PRI) — research agents alongside runtime iterations.

A runtime iteration spends most of its wall clock on mechanical work an LLM
does not need to babysit: compile, benchmark A/B cycles, lm_eval, server
restarts. The PRI uses that window: while runtime iteration V{N} runs, a swarm
of research agents (different personas, so they explore different lenses)
reads the whole campaign history and proposes further optimizations. They
sync through a moderator, critique each other, and produce an adjudicated
proposal set that the NEXT iteration's investigation consumes.

Protocol (one PRI per runtime iteration, fully self-contained)::

    R1: N persona agents in parallel -> agents/A{i}_r1.md
    SYNC1: moderator merges/dedupes -> sync/digest_r1.md
    R2: critique + refine + rank     -> agents/A{i}_r2.md
    MERGE: moderator union set       -> proposals_draft.md
    GATE: wait for V{N}/summary.txt  (the iteration's completion marker)
    ADJUDICATE: read V{N}'s results  -> proposals_final.md + proposals.json
    HANDOFF: append a pointer block to V{N}/summary.txt so the next
             investigation reads the proposals first

Enabled by CodeGenConfig::

    "enable_parallel_research_iteration": true,   # default
    "parallel_research_iteration_agent_count": 4  # default

run_code_gen.launch() spawns this module detached (``python3 -m
llm.auto_code_gen.parallel_research_iteration --orchestrate ...``) right
before the bench step of each iteration; the module terminates on its own
after adjudication (or after the gate timeout).

Guardrails enforced in every prompt: research agents are READ-ONLY and
GPU-free (the benchmark owns the GPUs), never build (a wedged HIP-clang
compile once cost 8h), and write only inside their research directory.
Their output is advisory: the investigation still validates before adopting.
"""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])

R1_TIMEOUT_S = 45 * 60
R2_TIMEOUT_S = 45 * 60
MOD_TIMEOUT_S = 30 * 60
ADJ_TIMEOUT_S = 40 * 60
GATE_POLL_S = 60
GATE_MAX_WAIT_S = 4 * 3600

# Persona pool (first N are used, N = parallel_research_iteration_agent_count).
# Each lens is deliberately different so the swarm converges on a broad,
# well-vetted idea set instead of N copies of the same idea.
PERSONAS = [
    ("kernel-micro",
     "Kernel-level micro-optimization specialist: launch overhead, kernel "
     "fusion, sorting/topk chains, memory access patterns in the model's "
     "hot kernel pipeline. You know a bad kernel TU can wedge the compiler "
     "for hours — never propose anything that risks a compile pathology "
     "without a concrete mitigation (e.g. a dedicated non-templated TU).",
     "Start from the previous iteration's analysis/ microbench results, then "
     "the kernels themselves in the target repo's csrc/ and any framework "
     "packages under the benchmark venv's site-packages."),
    ("scheduling-overlap",
     "Execution-scheduling specialist: stream overlap, CUDA/ROCm graph "
     "boundaries, aux-stream event syncs, allreduce/collective stalls, "
     "draft/target overlap in speculative decoding, prefill/decode "
     "interleave, server-level scheduling knobs for the campaign's batch "
     "size.",
     "Start from the earlier iterations' trace analyses and the model "
     "forward in the target repo, then the worker/scheduler code."),
    ("numerics-spec-decode",
     "Numerics & decoding specialist: quantization paths and their fidelity "
     "costs, precision/perf tradeoffs, and — if the workload uses "
     "speculative decoding — acceptance-length levers, verification cost, "
     "draft/target interactions. You know acceptance-length RNG across "
     "server restarts is a measured noise source — proposals must survive "
     "the campaign's stabilized-means protocol.",
     "Start from the bench result JSONs (acceptance length, ITL/TPOT "
     "distributions) in the recent runtime/V*/ dirs, then the quant code "
     "under the target repo's model_executor layers."),
    ("skeptic-red-team",
     "Red-team specialist: attack the current patch and the obvious next "
     "levers. Hunt correctness holes, performance traps, hidden regressions, "
     "and invalid assumptions in prior measurements — dead code that passes "
     "unit tests but never runs on the serving path is exactly your genre. "
     "Your mission is to prevent wasted benchmark runs AND to find the "
     "non-obvious levers everyone else overlooks.",
     "Start by reading the promoted patch itself critically, then diff it "
     "against what the summaries claim, then look where nobody looked "
     "(frontend, tokenizer/detokenizer path, logging overhead, scheduler "
     "polling)."),
    ("memory-bandwidth",
     "Memory-hierarchy specialist: weight residency and prefetch, L2 cache "
     "behavior across the layer pipeline, paging/allocator behavior at the "
     "campaign's batch size, KV-cache layout and access patterns, and which "
     "kernels are actually at the bandwidth floor vs launch-bound.",
     "Start from the trace/kernel analyses in the history to classify each "
     "hot kernel as BW-bound or latency-bound, then the memory traffic each "
     "proposed change adds or removes."),
    ("system-serving",
     "Serving-stack specialist: everything around the model forward — "
     "scheduler loops and polling, tokenizer/detokenizer and stop-string "
     "checking, output processing and streaming, request handling overhead, "
     "logging, and async-output paths. At tiny batch sizes these can be a "
     "measurable slice of the step.",
     "Start from the target repo's engine/serving entry points and the "
     "benchmark's client/server logs (timestamps in stdout logs reveal "
     "gaps between GPU work)."),
]


def pri_research_dir(output_dir: str, iteration: int) -> str:
    return os.path.join(output_dir, "research", "RI_over_V{}".format(iteration))


def pri_proposals_file(output_dir: str, iteration: int):
    """Path of iteration ``iteration``'s PRI proposals, or None if absent.

    Used by run_code_gen to feed the NEXT iteration's investigation prompt.
    """
    path = os.path.join(pri_research_dir(output_dir, iteration),
                        "proposals_final.md")
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    return None


def launch_parallel_research_iteration(config, iteration: int):
    """Spawn the PRI orchestrator detached for runtime iteration ``iteration``.

    Called by run_runtime_iterations right before the bench step: the
    compile+bench+lm_eval+investigation window is the idle-AI time the PRI
    fills. Returns the research dir, or None when a PRI already covers this
    iteration (resume) or the flag is off.
    """
    if not getattr(config, "enable_parallel_research_iteration", False):
        return None
    agent_count = getattr(config, "parallel_research_iteration_agent_count", 4)

    research_dir = pri_research_dir(config.output_dir, iteration)
    status_file = os.path.join(research_dir, "status.json")
    if os.path.isfile(status_file):
        # A PRI already ran (or is running) for this iteration — resume
        # logic inside the orchestrator picks up where it left off.
        print("  PRI: research iteration for V{} already present, not "
              "relaunching ({})".format(iteration, status_file))
        return None

    os.makedirs(research_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "llm.auto_code_gen.parallel_research_iteration",
           "--orchestrate",
           "--output-dir", config.output_dir,
           "--iteration", str(iteration),
           "--agent-count", str(agent_count)]
    agent_config_path = getattr(config, "agent_config_path", "")
    if agent_config_path:
        cmd += ["--agent-config", agent_config_path]

    log_path = os.path.join(research_dir, "orchestrator.log")
    log_file = open(log_path, "a")
    try:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    finally:
        log_file.close()
    with open(os.path.join(research_dir, "orchestrator.pid"), "w") as pf:
        pf.write(str(proc.pid) + "\n")
    print("  PRI: parallel research iteration launched over V{} "
          "({} agents) -> {}".format(iteration, agent_count, research_dir))
    return research_dir


# ---------------------------------------------------------------------------
# Prompt construction (grounding pack + rules are shared by every agent)
# ---------------------------------------------------------------------------

def _grounding(output_dir: str, iteration: int) -> str:
    prev = iteration - 1
    runtime = os.path.join(output_dir, "runtime")
    return f"""CONTEXT — read these files first, in this order (absolute paths;
skip any that do not exist, but 1-3 are essential):
1. {runtime}/iterations_history.txt  (full campaign history: all learnings,
   decision tables, rejected directions, root causes — your first act is to
   extract the current measured state from it: per-step time, TTFT/TPOT/ITL,
   acceptance/accuracy numbers, per-kernel timings)
2. {runtime}/summary_all.txt
3. {runtime}/V{prev}/summary.txt  (the currently-promoted state — read any
   ADDENDUM sections too; they are newer than the body)
4. {runtime}/V{prev}/summary_draft_R0.txt and {runtime}/V{prev}/analysis/
   (the previous investigation's decision table, microbench results and
   bench scripts)
5. Current promoted patch (what is already implemented):
   {runtime}/V{prev}/patch.patch
6. V{iteration} in-flight results (READ-ONLY, partial — the bench may still
   be running): {runtime}/V{iteration}/
7. Code under study (READ-ONLY): the target source tree, and any framework
   packages under the benchmark venv's site-packages (kernels often live
   there, outside the repo).
"""


def _rules(research_dir: str) -> str:
    return f"""HARD RULES (violating these poisons the running benchmark campaign):
- You are READ-ONLY everywhere except your own output file in {research_dir}/.
- NEVER modify the target source tree (no edits, no git checkout/reset/apply/
  stash, no cmake/ninja/build, no pip). NEVER touch the campaign's runtime/
  artifacts — reads only.
- NO GPU work of any kind (no torch CUDA/ROCm runs, no profilers, no python
  that initializes a GPU): the campaign's benchmark is running on this
  machine's GPUs right now, and other GPUs may belong to other users.
  Analysis only — empirical validation happens later in the investigation.
- NEVER launch servers, NEVER touch other users' processes.
- Bash is allowed for READ-ONLY inspection only (ls, cat, grep, sed -n, head,
  tail, wc, git diff/log/show/status, python -c for pure-CPU text analysis).
- Time-box: aim to finish this round in about 25-30 minutes of work.
"""

SCHEMA = """Every proposal must use EXACTLY this schema:

### P{agent}-{k}: {short title}
- Lever class: kernel | dispatch | scheduling | numerics | spec-decode | memory | serving | config
- Mechanism: the exact code change (files, functions, what moves where)
- Expected effect: TPOT -x% and/or TTFT -y% — quantitative reasoning grounded
  in measured data from the history (cite the numbers you build on)
- Evidence: which measurement / trace / code analysis supports this
- Risks: correctness / RNG-order / compile-pathology / regression risks, each
  with a mitigation
- Validation plan: the CHEAPEST experiment that would kill or confirm this
  (a microbench design with exact shapes/commands the investigation can run)
- Disable switch: proposed VLLM_DISABLE_* env var name
- Confidence: high | medium | low
"""


def _r1_prompt(i: int, persona, research_dir: str, output_dir: str,
               iteration: int) -> str:
    name, desc, doors = persona
    return f"""You are agent A{i} of a multi-agent parallel research swarm (PRI) running
ALONGSIDE runtime iteration V{iteration} of this code-gen campaign. Each
agent has a different lens so the swarm converges on a broad, well-vetted
idea set. Your persona: {name} — {desc}

Suggested entry points for your lens: {doors}

{_grounding(output_dir, iteration)}
MISSION (round 1 of 2 — private work, no visibility into other agents yet):
Through your persona's lens, propose 3-5 NEW optimization ideas to improve
the campaign's target metrics (the workload, its batch size and the metrics
that matter are described in the history — extract them first) — building on
ALL previous learnings. Ideas must NOT re-litigate a direction the history
already rejected unless you bring genuinely new evidence or a materially
different mechanism, and must NOT duplicate what the promoted patch already
implements. The optimization currently being benched in V{iteration} may
land or fail — cover both cases where relevant. Ground every estimate in
measured numbers from the history.

{_rules(research_dir)}
OUTPUT: write your round-1 output to {research_dir}/agents/A{i}_r1.md
(markdown, at most ~350 lines): a short "state of the world as I see it"
(<=15 lines, your own synthesis), then your 3-5 proposals.

{SCHEMA}
Then STOP — the orchestrator collects all agents' outputs, merges them into
a shared digest, and round 2 will critique and rank them.
"""


def _r2_prompt(i: int, persona, digest_path: str, research_dir: str,
               output_dir: str, iteration: int) -> str:
    name, desc, _doors = persona
    return f"""You are agent A{i} (persona: {name}) of the parallel research swarm (PRI),
round 2 of 2. Round 1 is complete: all agents' proposals were merged into a
shared digest by a moderator.

Read the digest FIRST: {digest_path}
Then re-read any history/code files you need (context list repeated below).

MISSION (round 2 — converge):
1. CRITIQUE at least 2 proposals from OTHER agents (name them by ID): find
   the flaw, the hidden cost, the invalidated assumption.
2. STEELMAN one proposal you initially disagree with: argue its strongest
   case before judging it.
3. REFINE your own round-1 ideas in light of the digest (kill your weakest,
   fix what others would attack, merge where a merge is stronger).
4. RANK the full digest set (all proposals, not just yours) from your
   persona's viewpoint: which 3 should the next runtime iteration's
   investigation validate FIRST, and why.

{_grounding(output_dir, iteration)}
{_rules(research_dir)}
OUTPUT: write your round-2 output to {research_dir}/agents/A{i}_r2.md
(markdown, at most ~300 lines): your critiques, your steelman, your refined
proposals (full schema below for new/changed ones), and your personal ranking
of the digest's top-3 with one-line justifications.

{SCHEMA}
"""


def _mod1_prompt(research_dir: str, output_dir: str, agent_ids) -> str:
    listing = "\n".join(
        "- {}/agents/A{}_r1.md  ({})".format(research_dir, i, PERSONAS[i - 1][0])
        for i in agent_ids)
    return f"""You are the moderator of a parallel research swarm (PRI) for this code-gen
campaign. Round 1 is complete.

Read ALL round-1 outputs:
{listing}

Also read {output_dir}/runtime/iterations_history.txt (the rejected-directions
decision tables matter) and the previous iteration's summary.

Produce the round-1 SYNC DIGEST at {research_dir}/sync/digest_r1.md:
1. A merged, DEDUPLICATED catalog of all proposals: merge overlapping ideas
   into one entry (list originating agent IDs), keep distinct ideas separate.
   Preserve each proposal's schema fields (mechanism, expected effect,
   evidence, risks, validation plan, disable switch, confidence).
2. A conflicts-with-history section: flag any proposal that re-litigates a
   rejected direction or contradicts a measured learning (cite it).
3. An overlaps-and-complements section: which proposals could combine.
4. Open questions the round-2 agents should attack.
Number every entry P1..Pk (stable IDs for round 2).
{_rules(research_dir)}
OUTPUT: {research_dir}/sync/digest_r1.md (markdown, at most ~450 lines)."""


def _merge_prompt(research_dir: str, agent_ids) -> str:
    return f"""You are the moderator of the parallel research swarm (PRI), final merge
phase. Round 2 (critique + ranking) is complete. Read:
- {research_dir}/sync/digest_r1.md  (the round-1 merged catalog, P1..Pk)
- All round-2 outputs: {research_dir}/agents/A{{1..{max(agent_ids)}}}_r2.md

Produce the DRAFT PROPOSAL SET at {research_dir}/proposals_draft.md:
1. The surviving union of proposals after round-2 critiques (drop entries
   that were killed by critique unless a defense survived; merge where
   round-2 agents merged; attribute merges and kills to the critics).
2. Every surviving entry keeps the FULL schema (mechanism, expected effect,
   evidence, risks, validation plan, disable switch, confidence).
3. A preliminary cross-vote ranking: aggregate the agents' top-3 votes
   (Borda-style: 3 points for a first choice, 2 for second, 1 for third).
4. A one-paragraph "recommended sequence for the next investigation".
{_rules(research_dir)}
OUTPUT: {research_dir}/proposals_draft.md (markdown, at most ~450 lines)."""


def _adjudicate_prompt(research_dir: str, output_dir: str, iteration: int) -> str:
    runtime = os.path.join(output_dir, "runtime")
    return f"""You are the adjudicator of the parallel research swarm (PRI) for this
code-gen campaign. Runtime iteration V{iteration} has now COMPLETED and its
results are final. Your job: read the fresh results, then finalize the
research proposal set that the NEXT runtime iteration's investigation will
consume.

Read, in this order:
1. {research_dir}/proposals_draft.md  (the draft proposal set from the swarm)
2. {runtime}/V{iteration}/summary.txt  (the iteration's final summary —
   bench verdict, lm_eval, what was promoted)
3. Everything V{iteration} produced: list {runtime}/V{iteration}/ and read
   the result artifacts (bench result JSONs, baseline results, perf
   comparison files, lm_eval results, summary drafts, patch drafts —
   read what exists).
4. Re-read {runtime}/iterations_history.txt if you need prior numbers.

For EVERY draft proposal, assign a post-iteration status:
- KILLED: the results invalidated its premise (say exactly which result)
- WEAKENED: partially undercut (say which)
- VALIDATED: premise confirmed by the numbers (say which)
- STRENGTHENED: the results opened a new door for it (say which)
- UNAFFECTED

Then produce the FINAL outputs:
1. {research_dir}/proposals_final.md — the finalized, re-ranked proposal set
   (surviving entries in full schema + post-iteration status + final ranking
   + a "RECOMMENDED NEXT LEVER" section naming the single best next
   investigation target and why). At most ~400 lines.
2. {research_dir}/proposals.json — machine-readable: a JSON array of objects
   with fields: id, title, lever_class, mechanism_summary, expected_effect,
   post_iteration_status, ranking, confidence, validation_plan_summary,
   disable_env_var.
{_rules(research_dir)}
OUTPUT: the two files above. Be decisive — the next investigation agent
will read proposals_final.md FIRST when choosing what to implement."""


def _handoff_block(research_dir_rel: str, timestamp: str) -> str:
    return """

================================================================================
PARALLEL RESEARCH ITERATION (PRI) HANDOFF — for the NEXT runtime iteration's
investigation (recorded {t} by the PRI orchestrator)
================================================================================
While this iteration ran, a research swarm (multiple personas; see
{rr}/config.json) worked in two rounds with a moderator sync, then
adjudicated against this iteration's final results. Its output — a ranked
set of pre-researched optimization proposals, each with mechanism, expected
effect, evidence, risks, a cheap validation plan and a proposed disable env
var — is at:
  {rr}/proposals_final.md   (human-readable, READ THIS FIRST)
  {rr}/proposals.json       (machine-readable)
Round artifacts: {rr}/agents/ and {rr}/sync/.
INVESTIGATION INSTRUCTIONS: treat these as strong candidates, not decisions:
run each candidate's cheap validation plan before committing; pick the best
surviving one; keep the campaign rules (own disable env var per improvement,
patches compound). Do NOT re-implement something this iteration already
promoted.
================================================================================
""".format(t=timestamp, rr=research_dir_rel)


# ---------------------------------------------------------------------------
# Single-agent runner mode (spawned by the orchestrator for each call)
# ---------------------------------------------------------------------------

async def _run_agent_session(name: str, prompt_file: str,
                             agent_config_path: str) -> None:
    from common.claude_utils import PipelineStep
    from llm.auto_code_gen.agents import agent_run, resolve_agent_config

    kwargs = {"cwd": REPO_ROOT}
    if agent_config_path:
        kwargs["agent_config_path"] = agent_config_path
    cfg = resolve_agent_config(**kwargs)
    with open(prompt_file) as f:
        prompt = f.read()
    await agent_run(cfg, [PipelineStep(name=name, prompt=prompt)])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print("[{}] {}".format(datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def _status(research_dir: str, phase: str, extra=None) -> None:
    st = {"phase": phase,
          "time": datetime.now().isoformat(timespec="seconds")}
    if extra:
        st.update(extra)
    with open(os.path.join(research_dir, "status.json"), "w") as f:
        json.dump(st, f, indent=2)


def _write_prompt(research_dir: str, fname: str, text: str) -> str:
    d = os.path.join(research_dir, "prompts")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, fname)
    with open(path, "w") as f:
        f.write(text)
    return path


async def _run_sub(name: str, prompt_path: str, log_path: str, timeout: int,
                   agent_config_path: str) -> bool:
    """Run one agent runner subprocess; kill its process group on timeout."""
    t0 = time.time()
    cmd = [sys.executable, "-m", "llm.auto_code_gen.parallel_research_iteration",
           "--run-agent", name, "--prompt-file", prompt_path]
    if agent_config_path:
        cmd += ["--agent-config", agent_config_path]
    with open(log_path, "wb") as lf:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=lf, stderr=asyncio.subprocess.STDOUT,
            cwd=REPO_ROOT, start_new_session=True)
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=timeout)
            _log("{}: exited rc={} after {:.0f} min".format(
                name, rc, (time.time() - t0) / 60))
            return rc == 0
        except asyncio.TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
            _log("{}: TIMED OUT after {}s (process group killed)".format(
                name, timeout))
            return False


async def orchestrate(output_dir: str, iteration: int, agent_count: int,
                      agent_config_path: str) -> None:
    research_dir = pri_research_dir(output_dir, iteration)
    os.makedirs(os.path.join(research_dir, "agents"), exist_ok=True)
    os.makedirs(os.path.join(research_dir, "sync"), exist_ok=True)

    personas = PERSONAS[:agent_count]
    if agent_count > len(PERSONAS):
        _log("WARNING: agent_count {} exceeds persona pool {}; using {}".format(
            agent_count, len(PERSONAS), len(PERSONAS)))
        agent_count = len(PERSONAS)
    agent_ids = list(range(1, agent_count + 1))

    if not os.path.exists(os.path.join(research_dir, "config.json")):
        with open(os.path.join(research_dir, "config.json"), "w") as f:
            json.dump({"num_agents": agent_count,
                       "personas": [p[0] for p in personas],
                       "rounds": 2, "iteration": iteration,
                       "agent_config": agent_config_path or "(default)",
                       "started": datetime.now().isoformat()}, f, indent=2)

    def afile(i: int, r: int) -> str:
        return os.path.join(research_dir, "agents", "A{}_r{}.md".format(i, r))

    # ---- Round 1 ----
    if all(os.path.exists(afile(i, 1)) for i in agent_ids):
        _log("R1 outputs already present — skipping round 1")
        have = agent_ids
    else:
        _status(research_dir, "R1_running")
        _log("PRI over V{}: ROUND 1 ({} agents in parallel)".format(
            iteration, agent_count))
        t0 = time.time()
        await asyncio.gather(*[
            _run_sub("PRI_A{}_R1".format(i),
                     _write_prompt(research_dir, "A{}_r1.txt".format(i),
                                   _r1_prompt(i, personas[i - 1], research_dir,
                                              output_dir, iteration)),
                     os.path.join(research_dir, "agents", "A{}_r1.log".format(i)),
                     R1_TIMEOUT_S, agent_config_path)
            for i in agent_ids])
        have = [i for i in agent_ids if os.path.exists(afile(i, 1))]
        _log("R1 done in {:.0f} min: outputs present={}".format(
            (time.time() - t0) / 60, have))
        if not have:
            _status(research_dir, "FAILED", {"phase_detail": "no R1 outputs"})
            return
        if len(have) < agent_count:
            _log("WARNING: proceeding with {}/{} R1 outputs".format(
                len(have), agent_count))

    # ---- SYNC 1 (moderator) ----
    digest = os.path.join(research_dir, "sync", "digest_r1.md")
    if os.path.exists(digest):
        _log("digest_r1.md already present — skipping SYNC 1")
    else:
        _status(research_dir, "SYNC1_running")
        _log("SYNC 1: moderator building digest")
        t0 = time.time()
        await _run_sub("PRI_MOD_R1",
                       _write_prompt(research_dir, "mod_r1.txt",
                                     _mod1_prompt(research_dir, output_dir, have)),
                       os.path.join(research_dir, "sync", "moderator_r1.log"),
                       MOD_TIMEOUT_S, agent_config_path)
        if not os.path.exists(digest):
            _status(research_dir, "FAILED", {"phase_detail": "no digest"})
            return
        _log("SYNC 1 done in {:.0f} min".format((time.time() - t0) / 60))

    # ---- Round 2 ----
    if all(os.path.exists(afile(i, 2)) for i in have):
        _log("R2 outputs already present — skipping round 2")
    else:
        _status(research_dir, "R2_running")
        _log("ROUND 2 (critique + ranking, agents {})".format(have))
        t0 = time.time()
        await asyncio.gather(*[
            _run_sub("PRI_A{}_R2".format(i),
                     _write_prompt(research_dir, "A{}_r2.txt".format(i),
                                   _r2_prompt(i, personas[i - 1], digest,
                                              research_dir, output_dir,
                                              iteration)),
                     os.path.join(research_dir, "agents", "A{}_r2.log".format(i)),
                     R2_TIMEOUT_S, agent_config_path)
            for i in have])
        have2 = [i for i in have if os.path.exists(afile(i, 2))]
        _log("R2 done in {:.0f} min: outputs present={}".format(
            (time.time() - t0) / 60, have2))
        if not have2:
            _status(research_dir, "FAILED", {"phase_detail": "no R2 outputs"})
            return

    # ---- MERGE ----
    draft = os.path.join(research_dir, "proposals_draft.md")
    if os.path.exists(draft):
        _log("proposals_draft.md already present — skipping MERGE")
    else:
        _status(research_dir, "MERGE_running")
        _log("MERGE: moderator producing proposals_draft.md")
        t0 = time.time()
        await _run_sub("PRI_MOD_MERGE",
                       _write_prompt(research_dir, "mod_merge.txt",
                                     _merge_prompt(research_dir, have)),
                       os.path.join(research_dir, "sync", "moderator_merge.log"),
                       MOD_TIMEOUT_S, agent_config_path)
        if not os.path.exists(draft):
            _status(research_dir, "FAILED", {"phase_detail": "no draft proposals"})
            return
        _log("MERGE done in {:.0f} min".format((time.time() - t0) / 60))

    # ---- GATE: wait for the runtime iteration to complete ----
    iter_dir = os.path.join(output_dir, "runtime", "V{}".format(iteration))
    summary = os.path.join(iter_dir, "summary.txt")
    if os.path.exists(summary):
        _log("V{} summary already present — gate already open".format(iteration))
        gate_ok = True
    else:
        _status(research_dir, "GATE_waiting_for_V{}".format(iteration))
        _log("GATE: waiting for {}".format(summary))
        t0 = time.time()
        gate_ok = True
        while not os.path.exists(summary):
            if time.time() - t0 > GATE_MAX_WAIT_S:
                gate_ok = False
                break
            await asyncio.sleep(GATE_POLL_S)
        if gate_ok:
            _log("GATE passed after {:.0f} min: V{} summary exists".format(
                (time.time() - t0) / 60, iteration))
        else:
            _log("GATE timed out after 4h — adjudicating with in-flight "
                 "results anyway")

    # ---- ADJUDICATE ----
    final = os.path.join(research_dir, "proposals_final.md")
    if os.path.exists(final):
        _log("proposals_final.md already present — skipping ADJUDICATION")
    else:
        _status(research_dir, "ADJUDICATING")
        _log("ADJUDICATION: reading V{} results, finalizing proposals".format(
            iteration))
        t0 = time.time()
        await _run_sub("PRI_ADJUDICATE",
                       _write_prompt(research_dir, "adjudicate.txt",
                                     _adjudicate_prompt(research_dir, output_dir,
                                                        iteration)),
                       os.path.join(research_dir, "adjudicator.log"),
                       ADJ_TIMEOUT_S, agent_config_path)
        if not os.path.exists(final):
            _status(research_dir, "FAILED", {"phase_detail": "no final proposals"})
            return
        _log("ADJUDICATION done in {:.0f} min".format((time.time() - t0) / 60))

    # ---- Handoff pointer into the iteration summary ----
    if gate_ok and os.path.exists(summary):
        rel = os.path.relpath(research_dir, os.path.dirname(summary))
        with open(summary, "a") as f:
            f.write(_handoff_block(
                rel, datetime.now().strftime("%Y-%m-%d %H:%M")))
        _log("Handoff block appended to V{}/summary.txt".format(iteration))
    else:
        _log("Iteration summary missing — handoff block NOT appended "
             "(check manually)")

    _status(research_dir, "COMPLETE")
    _log("PRI over V{} COMPLETE".format(iteration))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orchestrate", action="store_true",
                    help="run the full PRI protocol for one iteration")
    ap.add_argument("--run-agent", metavar="NAME",
                    help="single-agent mode: run one agent session")
    ap.add_argument("--prompt-file")
    ap.add_argument("--output-dir")
    ap.add_argument("--iteration", type=int)
    ap.add_argument("--agent-count", type=int, default=4)
    ap.add_argument("--agent-config", default="")
    args = ap.parse_args()

    if args.run_agent:
        if not args.prompt_file:
            ap.error("--run-agent requires --prompt-file")
        asyncio.run(_run_agent_session(args.run_agent, args.prompt_file,
                                       args.agent_config))
        return

    if not args.orchestrate:
        ap.error("nothing to do: pass --orchestrate or --run-agent")
    if not args.output_dir or not args.iteration:
        ap.error("--orchestrate requires --output-dir and --iteration")

    research_dir = pri_research_dir(args.output_dir, args.iteration)
    os.makedirs(research_dir, exist_ok=True)
    # Own log: the parent pipeline only keeps a one-line launch note.
    log_file = open(os.path.join(research_dir, "orchestrator.log"), "a",
                    buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        asyncio.run(orchestrate(args.output_dir, args.iteration,
                                args.agent_count, args.agent_config))
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
