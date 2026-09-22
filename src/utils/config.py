# src/utils/config.py
from pydantic import BaseModel, Field
from typing import Optional, List
from omegaconf import OmegaConf

class ModelConfig(BaseModel):
    name_or_path: str = "microsoft/phi-2"          # or TinyLlama, Qwen2-0.5B, Gemma-2B, etc.
    max_seq_length: int = 2048
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True

class PeftConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

class TrainingConfig(BaseModel):
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    bf16: bool = True
    gradient_checkpointing: bool = True
    deepspeed: Optional[str] = None               # path to ds_config.json
    report_to: List[str] = ["mlflow", "wandb"]

class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1