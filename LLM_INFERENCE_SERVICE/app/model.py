from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

hf_model_name = "distilgpt2"
device = "cpu"

class ModelService():

    def __init__(self, hf_model_name, device):
        
        self.model = hf_model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model) # loading tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model) # loading model params
        self.model.to(self.device) # moving model to device CPU/GPU
        
        # used for inference/validation of model, 
        # its job is to manage behaviour during training and inference phase of model,
        # for example, 
        #   - change layer behaviour
        #   - dropout layer deactivated during inference 
        self.model.eval()

    def generate(self, prompt:str, max_tokens:int, temperature:float):

        inputs = self.tokenizer(prompt, return_tensors = 'pt')

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                new_max_tokens = max_tokens,
                temperature = temperature,
                do_sample = temperature > 0                
                )
            
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def __repr__(self):

        print(f"Model service ran for model name as {self.model} on {self.device} device.")