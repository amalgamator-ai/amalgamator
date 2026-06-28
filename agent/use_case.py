import os

from common.claude_utils import PipelineStep, claude_run

from agent.agent_prompts import (
    create_context_str,
    AnalyzeHermesFeaturesPrompt,
    AnalyzeOpenClawFeaturesPrompt,
    DetectPortabilityCandidatesPrompt,
    CodeTracePrompt,
    CodePortPlanPrompt,
    ReviewCodePortPlanPrompt,
    TestPlanPrompt,
    ReviewTestPlanPrompt,
    CodeGenPrompt,
    ReviewCodeGenPrompt,
    ApplyAndBuildPrompt,
    RunAndTestPrompt,
    InvestigateAndFixPrompt,
    FEATURE_HERMES_FILE,
    FEATURE_OPENCLAW_FILE,
    DETECTED_FEATURES_FILE,
    DETECTED_FEATURES_JSON,
    CODE_PORT_PLAN_FILE_PREFIX,
    CODE_GEN_FILE_PREFIX,
    TEST_PLAN_PREFIX,
    RUNTIME_FILE_PREFIX,
    RUNTIME_SUCCESS_FILE,
    RUNTIME_LOGS_DIR,
    code_trace_filename,
)


class FeaturePortingUseCase:
    """Use case for porting features from Hermes Agent to OpenClaw."""

    def create_context_str(self, config):
        return create_context_str(config)

    # -- Phase 0: Feature Detection ------------------------------------------

    def gen_feature_detection_steps(self, context, config):
        hermes_prompt = AnalyzeHermesFeaturesPrompt(
            context=context,
            output_file=FEATURE_HERMES_FILE,
        )
        openclaw_prompt = AnalyzeOpenClawFeaturesPrompt(
            context=context,
            output_file=FEATURE_OPENCLAW_FILE,
        )
        detect_prompt = DetectPortabilityCandidatesPrompt(
            context=context,
            hermes_features_file=FEATURE_HERMES_FILE,
            openclaw_features_file=FEATURE_OPENCLAW_FILE,
            report_file=DETECTED_FEATURES_FILE,
            json_file=DETECTED_FEATURES_JSON,
            max_features=config.max_features_to_detect,
        )

        steps = [
            PipelineStep(
                name="analyze_hermes_features",
                prompt=hermes_prompt.prompt(),
                output_files=[FEATURE_HERMES_FILE],
            ),
            PipelineStep(
                name="analyze_openclaw_features",
                prompt=openclaw_prompt.prompt(),
                output_files=[FEATURE_OPENCLAW_FILE],
            ),
            PipelineStep(
                name="detect_portability_candidates",
                prompt=detect_prompt.prompt(),
                output_files=[DETECTED_FEATURES_FILE, DETECTED_FEATURES_JSON],
            ),
        ]

        return steps

    # -- Phase 1: Code Traces ------------------------------------------------

    def gen_code_trace_steps(self, context, config, feature_description):
        hermes_prompt = CodeTracePrompt(
            context=context,
            project="hermes",
            project_dir=config.hermes_source_dir,
            feature_description=feature_description,
            output_file=code_trace_filename("hermes"),
        )
        openclaw_prompt = CodeTracePrompt(
            context=context,
            project="openclaw",
            project_dir=config.openclaw_source_dir,
            feature_description=feature_description,
            output_file=code_trace_filename("openclaw"),
        )

        code_trace_files = [
            hermes_prompt.output_file,
            openclaw_prompt.output_file,
        ]

        steps = [
            PipelineStep(
                name="code_trace_hermes",
                prompt=hermes_prompt.prompt(),
                output_files=[hermes_prompt.output_file],
            ),
            PipelineStep(
                name="code_trace_openclaw",
                prompt=openclaw_prompt.prompt(),
                output_files=[openclaw_prompt.output_file],
            ),
        ]

        return steps, code_trace_files

    # -- Phase 2: Code Port Plan Iterations -----------------------------------

    def gen_code_port_plan_iter_steps(
        self, context, config, code_trace_files,
        feature_description,
        prev_output_file, prev_output_summary_file, iteration,
    ):
        plan_prompt = CodePortPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            feature_description=feature_description,
            output_file="{}_V{}.txt".format(CODE_PORT_PLAN_FILE_PREFIX, iteration),
            output_summary_file="{}_summary_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, iteration,
            ),
            prev_output_file=prev_output_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=iteration,
        )

        review_prompt = ReviewCodePortPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            feature_description=feature_description,
            input_file=plan_prompt.output_file,
            input_summary_file=plan_prompt.output_summary_file,
            output_file="{}_review_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, iteration,
            ),
            output_summary_file="{}_review_summary_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, iteration,
            ),
            iteration=iteration,
        )

        steps = [
            PipelineStep(
                name="code_port_plan_V{}".format(iteration),
                prompt=plan_prompt.prompt(),
                output_files=[plan_prompt.output_file, plan_prompt.output_summary_file],
            ),
            PipelineStep(
                name="code_port_plan_review_V{}".format(iteration),
                prompt=review_prompt.prompt(),
                output_files=[review_prompt.output_file, review_prompt.output_summary_file],
            ),
        ]

        return steps, review_prompt

    # -- Phase 3: Test Plan Iterations ----------------------------------------

    def gen_test_plan_iter_steps(
        self, context, config, code_trace_files,
        code_port_plan_file, feature_description,
        prev_output_file, prev_output_summary_file, iteration,
    ):
        test_plan_prompt = TestPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            feature_description=feature_description,
            output_file="{}_V{}.txt".format(TEST_PLAN_PREFIX, iteration),
            output_summary_file="{}_summary_V{}.txt".format(
                TEST_PLAN_PREFIX, iteration,
            ),
            prev_output_file=prev_output_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=iteration,
        )

        review_prompt = ReviewTestPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            feature_description=feature_description,
            input_file=test_plan_prompt.output_file,
            input_summary_file=test_plan_prompt.output_summary_file,
            output_file="{}_review_V{}.txt".format(TEST_PLAN_PREFIX, iteration),
            output_summary_file="{}_review_summary_V{}.txt".format(
                TEST_PLAN_PREFIX, iteration,
            ),
            iteration=iteration,
        )

        steps = [
            PipelineStep(
                name="test_plan_V{}".format(iteration),
                prompt=test_plan_prompt.prompt(),
                output_files=[test_plan_prompt.output_file, test_plan_prompt.output_summary_file],
            ),
            PipelineStep(
                name="test_plan_review_V{}".format(iteration),
                prompt=review_prompt.prompt(),
                output_files=[review_prompt.output_file, review_prompt.output_summary_file],
            ),
        ]

        return steps, review_prompt

    # -- Phase 4: Code Gen Iterations -----------------------------------------

    def gen_code_gen_iter_steps(
        self, context, config, code_trace_files,
        code_port_plan_file, test_plan_file,
        feature_description,
        prev_output_patch_file, prev_output_summary_file, iteration,
    ):
        code_gen_prompt = CodeGenPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            test_plan_file=test_plan_file,
            feature_description=feature_description,
            build_command=config.build_command,
            test_command=config.test_command,
            output_patch_file="{}_V{}.patch".format(CODE_GEN_FILE_PREFIX, iteration),
            output_summary_file="{}_summary_V{}.txt".format(
                CODE_GEN_FILE_PREFIX, iteration,
            ),
            prev_output_patch_file=prev_output_patch_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=iteration,
        )

        code_review_prompt = ReviewCodeGenPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            test_plan_file=test_plan_file,
            feature_description=feature_description,
            input_patch_file=code_gen_prompt.output_patch_file,
            input_summary_file=code_gen_prompt.output_summary_file,
            output_patch_file="{}_review_V{}.patch".format(
                CODE_GEN_FILE_PREFIX, iteration,
            ),
            output_summary_file="{}_review_summary_V{}.txt".format(
                CODE_GEN_FILE_PREFIX, iteration,
            ),
            iteration=iteration,
        )

        steps = [
            PipelineStep(
                name="code_gen_V{}".format(iteration),
                prompt=code_gen_prompt.prompt(),
                output_files=[
                    code_gen_prompt.output_patch_file,
                    code_gen_prompt.output_summary_file,
                ],
            ),
            PipelineStep(
                name="code_gen_review_V{}".format(iteration),
                prompt=code_review_prompt.prompt(),
                output_files=[
                    code_review_prompt.output_patch_file,
                    code_review_prompt.output_summary_file,
                ],
            ),
        ]

        return steps, code_review_prompt

    # -- Phase 5: Runtime Iterations ------------------------------------------

    async def run_runtime_iterations(
        self, context, config, claude_config,
        code_trace_files, code_port_plan_file, test_plan_file,
        feature_description, resume=False,
    ):
        import re
        import time

        output_dir = config.output_dir
        max_runtime_iterations = config.num_runtime_iterations
        runtime_logs_dir = os.path.join(output_dir, RUNTIME_LOGS_DIR)
        os.makedirs(runtime_logs_dir, exist_ok=True)

        phase_start = time.time()
        all_step_timings = []
        runtime_succeeded = False
        runtime_iterations = 0

        if resume and os.path.isfile(os.path.join(output_dir, RUNTIME_SUCCESS_FILE)):
            print("Resume: runtime iterations already succeeded")
            return [{
                "name": "Runtime",
                "iterations": 0,
                "max_iterations": max_runtime_iterations,
                "converged": True,
                "duration": 0.0,
            }], []

        latest_patch = self._find_latest_patch(output_dir)
        if latest_patch is None:
            print("ERROR: No code patch found in {}. Cannot run runtime iterations.".format(
                output_dir
            ))
            return [{
                "name": "Runtime",
                "iterations": 0,
                "max_iterations": max_runtime_iterations,
                "converged": False,
                "duration": time.time() - phase_start,
            }], []

        prev_patch_file = latest_patch

        print("\n" + "=" * 80)
        print("PHASE 5: RUNTIME ITERATIONS")
        print("=" * 80)
        print("  Max iterations: {}".format(max_runtime_iterations))
        print("  Initial patch: {}".format(prev_patch_file))
        print("=" * 80 + "\n")

        for i in range(max_runtime_iterations):
            iteration = i + 1
            runtime_iterations = iteration

            print("\n--- Runtime Iteration {} / {} ---".format(
                iteration, max_runtime_iterations
            ))

            # Step 1: Apply patch and build
            apply_prompt = ApplyAndBuildPrompt(
                context=context,
                patch_file=prev_patch_file,
                build_command=config.build_command,
                status_file="runtime_apply_status_V{}.txt".format(iteration),
                iteration=iteration,
            )
            steps_apply = [
                PipelineStep(
                    name="apply_build_V{}".format(iteration),
                    prompt=apply_prompt.prompt(),
                    output_files=[apply_prompt.status_file],
                ),
            ]
            timings = await claude_run(claude_config, steps_apply)
            all_step_timings.extend(timings)

            status_path = os.path.join(output_dir, apply_prompt.status_file)
            if os.path.isfile(status_path):
                with open(status_path) as f:
                    status = f.read().strip()
                if status != "SUCCESS":
                    print("Runtime iteration {}: Build FAILED".format(iteration))

            # Step 2: Run tests
            run_prompt = RunAndTestPrompt(
                context=context,
                test_command=config.test_command,
                results_file=os.path.join(
                    RUNTIME_LOGS_DIR,
                    "runtime_test_results_V{}.txt".format(iteration),
                ),
                status_file=os.path.join(
                    RUNTIME_LOGS_DIR,
                    "runtime_test_status_V{}.txt".format(iteration),
                ),
                iteration=iteration,
            )
            steps_run = [
                PipelineStep(
                    name="run_test_V{}".format(iteration),
                    prompt=run_prompt.prompt(),
                ),
            ]
            timings = await claude_run(claude_config, steps_run)
            all_step_timings.extend(timings)

            # Step 3: Investigate and fix if needed
            investigate_prompt = InvestigateAndFixPrompt(
                context=context,
                code_trace_files=code_trace_files,
                code_port_plan_file=code_port_plan_file,
                test_plan_file=test_plan_file,
                feature_description=feature_description,
                prev_patch_file=prev_patch_file,
                runtime_logs_dir=RUNTIME_LOGS_DIR,
                build_command=config.build_command,
                test_command=config.test_command,
                output_patch_file="{}_V{}.patch".format(RUNTIME_FILE_PREFIX, iteration),
                output_summary_file="{}_summary_V{}.txt".format(
                    RUNTIME_FILE_PREFIX, iteration,
                ),
                success_file=RUNTIME_SUCCESS_FILE,
                iteration=iteration,
            )
            steps_investigate = [
                PipelineStep(
                    name="investigate_runtime_V{}".format(iteration),
                    prompt=investigate_prompt.prompt(),
                ),
            ]
            timings = await claude_run(claude_config, steps_investigate)
            all_step_timings.extend(timings)

            if os.path.isfile(os.path.join(output_dir, RUNTIME_SUCCESS_FILE)):
                print("\nRuntime iteration {}: SUCCESS!".format(iteration))
                runtime_succeeded = True
                break

            new_patch = "{}_V{}.patch".format(RUNTIME_FILE_PREFIX, iteration)
            if not os.path.isfile(os.path.join(output_dir, new_patch)):
                print("Runtime iteration {}: No new patch generated, stopping.".format(
                    iteration
                ))
                break

            prev_patch_file = new_patch

        return [{
            "name": "Runtime",
            "iterations": runtime_iterations,
            "max_iterations": max_runtime_iterations,
            "converged": runtime_succeeded,
            "duration": time.time() - phase_start,
        }], all_step_timings

    @staticmethod
    def _find_latest_patch(output_dir):
        import re
        runtime_pat = re.compile(r'^code_gen_runtime_V(\d+)\.patch$')
        codegen_pat = re.compile(r'^code_gen_review_V(\d+)\.patch$')

        latest_runtime = None
        latest_runtime_ver = 0
        latest_codegen = None
        latest_codegen_ver = 0

        for f in os.listdir(output_dir):
            m = runtime_pat.match(f)
            if m:
                ver = int(m.group(1))
                if ver > latest_runtime_ver:
                    latest_runtime_ver = ver
                    latest_runtime = f
            m = codegen_pat.match(f)
            if m:
                ver = int(m.group(1))
                if ver > latest_codegen_ver:
                    latest_codegen_ver = ver
                    latest_codegen = f

        if latest_runtime:
            return latest_runtime
        return latest_codegen
