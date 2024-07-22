from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import os
import json

class ModelHandler:
    def __init__(self, config):
        self.model_type = config['model_type']
        self.sbert_model = SentenceTransformer(config['sbert_model_name'])
        
        if self.model_type == 'gpt-4o-mini':
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.client = openai.OpenAI(api_key=openai.api_key)
            self.config = config
        elif self.model_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(config['llama_model_name'], token=os.getenv('HUGGINGFACE_TOKEN'))
            self.model = AutoModelForCausalLM.from_pretrained(config['llama_model_name'], token=os.getenv('HUGGINGFACE_TOKEN'))
        else:
            raise ValueError("Unsupported model type specified in config")

    def generate_embedding(self, text):
        return self.sbert_model.encode(text)

    def generate_response(self, prompt, system_prompt=None):
        if self.model_type == 'gpt-4o-mini':
            messages = [
                {"role": "system", "content": system_prompt or self.config.get('system_prompt', '')},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.config['gpt4_model_name'],
                messages=messages,
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
                top_p=self.config['top_p'],
                frequency_penalty=self.config['frequency_penalty'],
                presence_penalty=self.config['presence_penalty']
            )
            return response.choices[0].message.content
        elif self.model_type == 'llama':
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
