from llm.auto_profile_server_client.frameworks.sglang import SglangAdapter
from llm.auto_profile_server_client.frameworks.trt import TrtAdapter
from llm.auto_profile_server_client.frameworks.vllm import VllmAdapter

ADAPTERS = {
    "vllm": VllmAdapter,
    "sgl": SglangAdapter,
    "trt": TrtAdapter,
}


def get_adapter(framework_name: str):
    cls = ADAPTERS.get(framework_name)
    if cls is None:
        raise ValueError(
            f"Unknown framework: {framework_name!r}. Valid: {list(ADAPTERS)}"
        )
    return cls()
