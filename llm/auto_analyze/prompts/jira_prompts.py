import os
from dataclasses import dataclass
from typing import ClassVar

from llm.auto_analyze.configs.single_trace_config import PERF_ANALYSIS_SINGLE_FILE
from llm.auto_analyze.configs.cross_trace_config import CROSS_COMPARE_FILE

JIRA_TASKS_FILE = "jira_tasks_output.txt"


@dataclass
class JiraTasksPrompt:
    f"""Create JIRA tasks from analysis results.

    Supports two modes:
      (A) Single-trace: creates tasks from {PERF_ANALYSIS_SINGLE_FILE} improvement proposals
      (B) Cross-trace: creates tasks from {CROSS_COMPARE_FILE} improvement plan
    """

    results_dir: str
    mode: str  # "single" or "cross"
    run_params_file: str
    analysis_file: str
    output_file: str

    prompt_template: ClassVar[str] = """
[results_dir] = {results_dir}
[run_params_file] = {run_params_file}
[analysis_file] = {analysis_file}
[output_file] = {output_file}

{mode_context}

<jira_epyc> = "Auto generated tasks for vLLM improvement"
<jira_team_value> = INFERENG Runtime
<jira_activity_type> = "Learning & Enablement"

Read [run_params_file] to understand the analysis that was performed (model, GPU, execution parameters, frameworks involved).
Read [analysis_file] which contains the improvement proposals/plans.

The goal of this task is to create a set of JIRA tasks for each improvement proposal in [analysis_file]. Do the following and think hard:
- Use JIRA MCP for this task
- Analyze and understand in-depth the improvement proposals in [analysis_file]
- Verify that epyc <jira_epyc> exists. If it does not exist, then STOP and DO NOT continue this task.
- To represent the improvement proposals in JIRA, create a (master) task with sub-tasks, where each sub-task represents an individual improvement proposal.
    - Create a (master) task as follows:
        - The parent epyc of the master task must be <jira_epyc>.
        - The team value must be <jira_team_value>.
        - The activity type must be <jira_activity_type>.
        - Name the task using the model name, execution parameters, and a timestamp: tasks_for_[model]_[params]__TIME_[YYYY-MM-DD-HH-MM-SS]. Detect the parameters from [run_params_file].
        - Add a task description that describes:
            - High-level results of the improvement plan. Add important details to make it clear for both executives and programmers.
            - Info about the execution metadata from [run_params_file] (frameworks, GPU type, model, execution parameters).
            - Links to the analysis results in [results_dir].
        - For the links mention on what machine they reside: hostname and ip address.
    - For each specific improvement proposal, create a sub-task as follows:
        - The parent task is the master task from above.
        - The team value must be <jira_team_value>.
        - Name the task as: plan_[id]_[plan_topic] where [id] is the serial id and [plan_topic] is the topic.
        - Add a sub-task description that describes this specific plan. Make sure to include:
            - High-level description, including the general metrics of impact, difficulty and more.
            - Step-by-step guide with code snippets and maximum details per-step so that an expert programmer can execute on it (be consistent with what is inside [analysis_file])

- Make sure that all new JIRA tasks and sub-tasks created are completely new and are not overriding any existing tasks or sub-tasks. DO NOT modify anything existing in JIRA.
- Provide a summary of what was done and write it to [output_file].

"""

    SINGLE_CONTEXT: ClassVar[str] = f"""This is a SINGLE-TRACE analysis.
[analysis_file] is the {PERF_ANALYSIS_SINGLE_FILE} from a single framework trace analysis.
It contains improvement proposals (P1, P2, ...) for the analyzed framework.
Create one JIRA sub-task per improvement proposal."""

    CROSS_CONTEXT: ClassVar[str] = f"""This is a CROSS-TRACE comparison.
[analysis_file] is the {CROSS_COMPARE_FILE} from a cross-framework or cross-commit analysis.
It contains an improvement plan for the target framework based on comparison with other traces.
Create one JIRA sub-task per improvement plan item."""

    def prompt(self):
        mode_context = self.SINGLE_CONTEXT if self.mode == "single" else self.CROSS_CONTEXT

        return self.prompt_template.format(
            results_dir=self.results_dir,
            run_params_file=self.run_params_file,
            analysis_file=self.analysis_file,
            output_file=self.output_file,
            mode_context=mode_context,
        )
