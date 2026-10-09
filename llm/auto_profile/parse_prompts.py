from dataclasses import dataclass
from typing import ClassVar


@dataclass
class ParseResultsPrompt:
    results_dir: str
    output_dir: str
    prompt_template: ClassVar[str] = """

The goal of this task is to parse and compare benchmark and profiling results of 3 frameworks: vllm, sgl and trt.
For this task think hard.

IMPORTANT: All output files, helper scripts, generated code, and any other artifacts
must be written to <output_dir> or its subdirectories. Do not create files anywhere else.

<results_dir> = {results_dir} 
<output_dir> = {output_dir} 
<metadata_file> = run_metadata.txt
<test_id>=[model]-tp_$[num_gpus]-isl_[input_len]-osl_[output_len]
<test_id_with_batch>=<test_id>-b_[concurrency]
<full_test_id>=[framework]-<test_id_with_batch>

Inside <results_dir> directory there are 3 sub-directories: vllm, sgl and trt. Each such sub-directory holds results for a specific framework.

The structure of each sub-directory is as follows:
- Has a <metadata_file> file with the metadata of executing the framework. It has Docker image tag, GPU type and more.
- Each model tested has a directory, for example, HF nvidia/DeepSeek-R1-NVFP4 is represented as nvidia__DeepSeek-R1-NVFP4
- Inside each model’s directory, there is a sequence of test directories, where each test directory represents a specific test with specific parameters
- Each test directory name has the following format (that encodes test parameters): test-<full_test_id>
- Inside each test directory there are the following files (that we call <test_files>): 
	- bench-<full_test_id>.json file that has the benchmark results of running the test on the framework
	- run-log-<full_test_id>.txt file that has the run log of the execution
	- [OPTIONAL] run-log-profile-<full_test_id>.txt file that has the profile run log of the execution
	- [OPTIONAL] trace-<full_test_id>.nsys-rep file that has the NSYS profile results
	- [OPTIONAL] trace-<full_test_id>.sqlite file that is the SQLite version of the nsys-rep file	
- The contents of each bench-<full_test_id>.json are specific for each framework. We are interested in TPOT. If it is not presented directly, then compute it.
- Each of the format dirnames/filenames may have also "-mode_[mode]" at the end which is optional

<test_results_dir> = <output_dir>/test_results
<test_dir> = <test_results_dir>/<test_id_with_batch>

First, restructure the directories from <results_dir> into <test_results_dir>, so that they are organized as follows:
- Create <test_results_dir> (if it does not exist), and ensure it is empty
- For each <test_id_with_batch>, create <test_dir>, and inside this directory:
    - Create <test_dir>/vllm, <test_dir>/sgl and <test_dir>/trt, a directory for each framework
    - For each framework directory, copy all of the associated <test_files> into this directory.
    - Also, copy the associated <metadata_file> of this test into <test_dir>

Next, generate a python program that creates a sequence of summary comparison tables as follows:
- Create a single summary table for each <test_id> where the table has the following format:
    - Columns are frameworks names: vLLM, SGLang and TRT (from left to right)
    - Rows are batch sizes tested: from small batch size to the larger one.
    - Entries are TPOT results for each framework and batch size
    - For each table provide a header/description with:
        - <test_id> parameters used
        - “mode” used (if exists)
        - Docker image used
        - GPU type used
        - For each batch size tested, the full directory path to <test_dir>, sorted from small batch size to larger
- Create a txt file for each table, where the filename encodes the table <test_id>. Ensure the table is aligned properly, clear and concise.
- Create a single txt file that has all tables together. Ensure all tables are aligned properly.
- Create a PDF file with all of the tables.
- Store all results in <output_dir>

Now do the following:
- For each test case <test_id_with_batch> where vLLM is slower than other framework, generate the following outputs inside <test_results_dir>/<test_id_with_batch>:
    - For each framework involved, a single trace config json file for this framework. Based on the example file single_trace_config_example.json and based on the results in <test_results_dir>/<test_id_with_batch>/[framework].
        - Ensure the output_dir is analyze_[framework] inside the <test_results_dir>/<test_id_with_batch>
        - Get the framework source code path from source_codes.json
        - Get the framework commit id from metadata.txt. If does not exist, then try to extract it from the run log, and if not found, then default to HEAD.
        - Extract the run_command from the run log txt file and use it in the json config
        - For run log, prefer the non-profile log
        - For trace file, if it is NSYS, then prefer the *.sqlite version as the path (if exists).
        - If this is an NSYS trace, then make trace_gpu_focus on GPU 0
        - Ensure skip_perf_analysis is set to true (this is a cross framework comparison)
    - A cross trace config json file that compares all of the frameworks involved where the target framework is vllm. Based on the example file cross_trace_config_example.json and the previously generated single trace json configs. Ensure make_improvement_plan is set to true.

"""

    def prompt(self):
        return self.prompt_template.format(
            results_dir=self.results_dir,
            output_dir=self.output_dir,
        )
