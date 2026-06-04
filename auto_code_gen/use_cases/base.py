from abc import ABC, abstractmethod

from common.claude_utils import ClaudeConfig, PipelineStep


class UseCase(ABC):
    """Defines domain-specific prompt content for the code gen pipeline.

    The pipeline orchestrator (run_code_gen.py) calls these methods to get
    PipelineStep objects with use-case-specific prompt templates.  The
    orchestration flow (phases, iteration loops, convergence, resume, timing)
    stays identical for every use case.
    """

    @abstractmethod
    def create_context_str(self, claude_config: ClaudeConfig, config) -> str:
        """Build the <context> block injected into every prompt."""
        ...

    @abstractmethod
    def gen_code_trace_steps(
        self, context: str, config,
    ) -> tuple[list[PipelineStep], list[str]]:
        """Returns (pipeline_steps, code_trace_file_names)."""
        ...

    @abstractmethod
    def gen_code_port_plan_iter_steps(
        self, context: str, config, code_trace_files: list[str],
        prev_output_file, prev_output_summary_file, iteration: int,
    ) -> tuple[list[PipelineStep], object]:
        """One code-port-plan generate+review iteration.

        Returns (steps, review_prompt) where review_prompt has
        .output_file and .output_summary_file for the convergence check.
        """
        ...

    def skip_test_plan_phase(self, config) -> bool:
        """Whether to skip Phase 3 (test plan iterations).

        Return True when the code port plan already includes the test plan
        (e.g. bug fix combined mode).
        """
        return False

    @abstractmethod
    def gen_test_plan_iter_steps(
        self, context: str, config, code_trace_files: list[str],
        code_port_plan_file: str,
        prev_output_file, prev_output_summary_file, iteration: int,
    ) -> tuple[list[PipelineStep], object]:
        """One test-plan generate+review iteration."""
        ...

    @abstractmethod
    def gen_code_gen_iter_steps(
        self, context: str, config, code_trace_files: list[str],
        code_port_plan_file: str, test_plan_file: str,
        prev_output_patch_file, prev_output_summary_file, iteration: int,
    ) -> tuple[list[PipelineStep], object]:
        """One code-gen generate+review iteration."""
        ...

    @abstractmethod
    async def run_runtime_iterations(
        self, context: str, config, claude_config: ClaudeConfig,
        code_trace_files: list[str],
        code_port_plan_file: str, test_plan_file: str,
        resume: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """Phase 5: use-case-specific runtime iterations.

        For LLM framework: apply patch, run benchmark, investigate, fix, repeat.
        For bug fix: autonomous build-test-fix loop.

        Returns (phase_results, step_timings).
        """
        ...

    @abstractmethod
    def clear_target_cmd(self, config) -> dict:
        """Command dict to clear target repo before steps that modify it."""
        ...
