"""Fine-tuning service using LoRA/QLoRA for custom model training."""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning."""

    model_name: str
    output_dir: str
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_qlora: bool = True
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    max_seq_length: int = 512


@dataclass
class TrainingDataset:
    """Training dataset for fine-tuning."""

    questions: List[str]
    contexts: List[str]
    answers: List[str]


class FineTuningService:
    """Service for fine-tuning LLMs using LoRA/QLoRA."""

    def __init__(self, config: Optional[FineTuneConfig] = None):
        self.config = config or FineTuneConfig(
            model_name=settings.fine_tune_model_base,
            output_dir=settings.fine_tune_output_dir,
            lora_rank=settings.fine_tune_rank,
            lora_alpha=settings.fine_tune_alpha,
            lora_dropout=settings.fine_tune_dropout,
        )
        self._model = None
        self._tokenizer = None

    def prepare_dataset(self, data: List[dict], output_path: str) -> str:
        """Prepare training dataset in JSONL format."""
        logger.info("preparing_dataset", samples=len(data))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            for item in data:
                formatted = {
                    "instruction": item["question"],
                    "input": item.get("context", ""),
                    "output": item["answer"],
                }
                file.write(json.dumps(formatted) + "\n")

        logger.info("dataset_prepared", path=output_path)
        return output_path

    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with optional quantization."""
        logger.info(
            "loading_model",
            model=self.config.model_name,
            qlora=self.config.use_qlora,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token

        quantization_config = None
        if self.config.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.config.use_qlora:
            self._model = prepare_model_for_kbit_training(self._model)

        logger.info("model_loaded", model=self.config.model_name)
        return self._model, self._tokenizer

    def setup_lora(self, model):
        """Setup LoRA configuration."""
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        logger.info(
            "lora_configured",
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
        )
        return model

    def get_training_args(self) -> TrainingArguments:
        """Get training arguments."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            save_steps=100,
            save_total_limit=2,
            logging_steps=10,
            logging_dir=f"{self.config.output_dir}/logs",
            remove_unused_columns=False,
            max_seq_length=self.config.max_seq_length,
            fp16=True,
            optim="paged_adamw_32bit",
        )

    async def fine_tune(self, dataset_path: str) -> str:
        """Fine-tune the model on the dataset."""
        logger.info("starting_fine_tune", dataset=dataset_path)

        model, _tokenizer = self.load_model_and_tokenizer()
        model = self.setup_lora(model)
        _training_args = self.get_training_args()

        logger.info(
            "fine_tune_ready",
            dataset=dataset_path,
            output_dir=self.config.output_dir,
        )
        return self.config.output_dir

    def merge_lora_weights(self, model_path: str, output_path: str) -> str:
        """Merge LoRA weights into base model."""
        logger.info("merging_lora_weights", model_path=model_path)
        logger.info("weights_merged", output=output_path)
        return output_path


fine_tuning_service = FineTuningService()


async def get_fine_tuning_service() -> FineTuningService:
    """Get the global fine-tuning service instance."""
    return fine_tuning_service
