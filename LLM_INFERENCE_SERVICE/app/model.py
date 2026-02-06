from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
import os
import re
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

logger = logging.getLogger("LLM INFERENCE SERVICE")

try:
    import psutil
except Exception:
    psutil = None


class ModelService:

    def __init__(self, hf_model_name: str, device: str):

        self.model_name = hf_model_name
        self.device = device

        # Tokenizer (chat template aware)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        )

        # INT8 dynamic quantization (CPU-optimized)
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        self.model = model.to(self.device)
        self.model.eval()

        # Padding safety
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Chat messages (NOT string prompts)
    # ------------------------------------------------------------------
    def build_messages(self, user_prompt: str):
        return [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant. "
                    "You must provide only the final answer. "
                    "Do not include any reasoning, thinking, analysis, or internal deliberation. "
                    "Do not use tags like <think>. "
                    "Respond with the answer only."
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    
    # ------------------------------------------------------------------
    # Minimal, safe post-processing
    # ------------------------------------------------------------------
    def extract_answer_only(self, text: str) -> str:
        # Take first paragraph only (safety guard)
        return text.split("</think>", 1)[-1].strip()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, prompt: str, max_tokens: int, temperature: float):

        messages = self.build_messages(prompt)

        start = time.perf_counter()

        mem_before_mb = None
        process = None
        if psutil is not None:
            process = psutil.Process(os.getpid())
            mem_before_mb = process.memory_info().rss / (1024 * 1024)

        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,   # CRITICAL
            return_tensors="pt"
        ).to(self.device)

        prompt_tokens = inputs.shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True
            )

        # Decode ONLY newly generated tokens
        decoded_raw = self.tokenizer.decode(
            output_ids[0][prompt_tokens:],
            skip_special_tokens=True
        )

        decoded = self.extract_answer_only(decoded_raw)

        elapsed_ms = (time.perf_counter() - start) * 1000

        mem_after_mb = None
        mem_delta_mb = None
        if psutil is not None:
            mem_after_mb = process.memory_info().rss / (1024 * 1024)
            mem_delta_mb = mem_after_mb - mem_before_mb

        logger.info(
            "Generation stats (prompt_tokens=%d, new_tokens=%d, elapsed_ms=%.2f, rss_delta_mb=%s)",
            prompt_tokens,
            len(output_ids[0]) - prompt_tokens,
            elapsed_ms,
            f"{mem_delta_mb:.2f}" if mem_delta_mb is not None else "N/A",
        )
        logger.info("=="*100)
        logger.info("Raw output: %s", decoded_raw)
        logger.info("=="*100)
        logger.info("Processed output: %s", decoded)

        return decoded

    def __repr__(self):
        return f"ModelService(model={self.model_name}, device={self.device})"
