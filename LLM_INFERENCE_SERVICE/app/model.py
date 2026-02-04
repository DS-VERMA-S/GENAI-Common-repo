from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

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

    # def build_prompt(self, user_prompt: str) -> str:
    #         return f"""
    #             You are an AI assistant.

    #             Rules:
    #             - Answer the question directly.
    #             - Do NOT restate the question.
    #             - Do NOT ask follow-up questions.
    #             - Do NOT include markdown.
    #             - Provide exactly one paragraph.
    #             - Stop after answering.

    #             Question:
    #             {user_prompt}

    #             Answer:
    #             """.strip()

    def build_prompt(self, user_prompt: str) -> str:
        return f"""
            Answer the following question in a single paragraph of plain text.

            Question:
            {user_prompt}

            Answer:
            """.strip()

    import re

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


    def hard_stop(self, text: str) -> str:
        STOP_STRINGS = [
            "Question:",
            "Answer:",
            "Rules:",
            "If you",
            "Now,",
        ]

        for s in STOP_STRINGS:
            if s in text:
                text = text.split(s, 1)[0]
        return text.strip()

    import re

    def sanitize_output(self, text: str) -> str:
        # Remove leftover instruction-like lines
        patterns = [
            r"If you don't know.*",
            r"Now, answer.*",
            r"The answer should.*",
            r"Provide.*paragraph.*",
            r"Rules:.*",
        ]

        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    def clean_output(self, text: str) -> str:
        # Remove anything before "Answer:"
        if "Answer:" in text:
            text = text.split("Answer:", 1)[-1]

        # Hard stop if model tries to continue conversation
        for stop_token in ["Question:", "User:", "###"]:
            if stop_token in text:
                text = text.split(stop_token, 1)[0]

        return text.strip()

    def enforce_word_limit(self, text: str, max_words: int) -> str:
        words = text.strip().split()
        return " ".join(words[:max_words])



    def generate(self, prompt: str, max_tokens: int, temperature: float):

        prompt = self.build_prompt(prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

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

        decoded = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        decoded = self.extract_answer_only(decoded)
        # decoded = self.enforce_word_limit(decoded, 100)
        return decoded

    def __repr__(self):

        print(f"Model service ran for model name as {self.model} on {self.device} device.")