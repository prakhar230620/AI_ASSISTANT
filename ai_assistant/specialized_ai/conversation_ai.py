# conversation_ai.py
from .base_ai import BaseAI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


class ConversationAI(BaseAI):
    def load_model(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)

    def process(self, input_data):
        input_ids = self.tokenizer.encode(input_data, return_tensors='pt').to(self.device)
        attention_mask = torch.ones(input_ids.shape, device=self.device)

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_capabilities(self):
        return ["text_conversation", "question_answering", "text_completion"]

    def fine_tune(self, dataset):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(3):  # You can adjust the number of epochs
            for batch in dataset:
                inputs = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(
                    self.device)
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch + 1} completed")


# Register the AI
from . import registry

registry.register("ConversationAI", ConversationAI)