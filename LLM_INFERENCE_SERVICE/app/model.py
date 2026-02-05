from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
import os
import torch
import re

logger = logging.getLogger("llm_inference_service")
try:
    import psutil
except Exception:  # pragma: no cover - fallback if psutil not installed
    psutil = None

class ModelService():

    def __init__(self, hf_model_name, device):
        
        self.model = hf_model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model) # loading tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=torch.float32) # loading model params
        self.model.to(self.device) # moving model to device CPU/GPU
        
        # used for inference/validation of model, 
        # its job is to manage behavior during training and inference phase of model,
        # for example, 
        #   - change layer behavior
        #   - dropout layer deactivated during inference 
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_prompt(self, user_prompt: str) -> str:
        return f"""
            Answer the following question in a single paragraph of plain text.

            Question:
            {user_prompt}

            Answer:
            """.strip()



    def extract_answer_only(self, text: str) -> str:
        # Remove everything before "Answer:"
        if "Answer:" in text:
            text = text.split("Answer:", 1)[-1]

        # Split into sentences / paragraphs
        # Keep only the FIRST paragraph-like chunk
        parts = re.split(r"\n\s*\n", text.strip())

        answer = parts[0]

        # Remove any trailing instruction-like fragments
        answer = re.sub(
            r"(please|answer|question|rule|markdown).*",
            "",
            answer,
            flags=re.IGNORECASE
        )

        # Normalize whitespace
        answer = re.sub(r"\s+", " ", answer).strip()

        # Final hard stop if model stops mid-answer
        answer = answer.split(".")
        answer.pop(-1)        

        return ".".join(answer) + "."

    def generate(self, prompt: str, max_tokens: int, temperature: float):

        prompt = self.build_prompt(prompt)

        start = time.perf_counter()
        mem_before_mb = None
        if psutil is not None:
            process = psutil.Process(os.getpid())
            mem_before_mb = process.memory_info().rss / (1024 * 1024)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        decoded_raw = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        total_tokens = output_ids[0].shape[0]
        new_tokens = max(total_tokens - prompt_tokens, 0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        mem_after_mb = None
        mem_delta_mb = None
        if psutil is not None:
            mem_after_mb = process.memory_info().rss / (1024 * 1024)
            mem_delta_mb = mem_after_mb - mem_before_mb

        decoded = self.extract_answer_only(decoded_raw)
        if psutil is not None:
            logger.info(
                "Generation stats (prompt_tokens=%d, total_tokens=%d, new_tokens=%d, elapsed_ms=%.2f, rss_before_mb=%.2f, rss_after_mb=%.2f, rss_delta_mb=%.2f)",
                prompt_tokens,
                total_tokens,
                new_tokens,
                elapsed_ms,
                mem_before_mb,
                mem_after_mb,
                mem_delta_mb,
            )
        else:
            logger.info(
                "Generation stats (prompt_tokens=%d, total_tokens=%d, new_tokens=%d, elapsed_ms=%.2f)",
                prompt_tokens,
                total_tokens,
                new_tokens,
                elapsed_ms,
            )
        logger.info("Raw output: %s", decoded_raw)
        logger.info("Processed output: %s", decoded)
        return decoded

    def __repr__(self):

        print(f"Model service ran for model name as {self.model} on {self.device} device.")
