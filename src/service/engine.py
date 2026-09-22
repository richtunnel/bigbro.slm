from vllm import LLM, SamplingParams
from src.utils.config import ServingConfig, ModelConfig

class SLMEngine:
    def __init__(self, model_cfg: ModelConfig, serving_cfg: ServingConfig):
        self.llm = LLM(
            model=model_cfg.name_or_path,           # or path to merged PEFT adapter
            tensor_parallel_size=serving_cfg.tensor_parallel_size,
            max_model_len=serving_cfg.max_model_len,
            gpu_memory_utilization=serving_cfg.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
        )

    def generate(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text for o in outputs]