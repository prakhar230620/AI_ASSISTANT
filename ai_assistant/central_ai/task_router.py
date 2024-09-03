# task_router.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

class IntentRecognizer(nn.Module):
    def __init__(self, num_intents):
        super(IntentRecognizer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_intents)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class TaskRouter:
    def __init__(self, central_ai, specialized_ais, num_intents=10):
        self.central_ai = central_ai
        self.specialized_ais = specialized_ais
        self.intent_recognizer = IntentRecognizer(num_intents)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.intent_clusters = None
        self.kmeans = KMeans(n_clusters=len(specialized_ais))
        self.intent_history = defaultdict(list)
        self.optimizer = torch.optim.Adam(self.intent_recognizer.parameters(), lr=1e-5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intent_recognizer.to(self.device)

    async def train_intent_recognizer(self, training_data):
        self.intent_recognizer.train()
        for epoch in range(3):
            total_loss = 0
            for text, intent in training_data:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = torch.tensor([intent]).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.intent_recognizer(**inputs)

                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(training_data)}")

    def cluster_intents(self, intent_embeddings):
        if intent_embeddings.shape[0] > 0:
            self.intent_clusters = self.kmeans.fit_predict(intent_embeddings)
        else:
            self.intent_clusters = None

    async def route_task(self, user_input):
        inputs = self.tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.intent_recognizer.eval()
        with torch.no_grad():
            intent_logits = self.intent_recognizer(**inputs)
            intent_embedding = F.softmax(intent_logits, dim=1).cpu().numpy()

        if self.intent_clusters is None or len(self.intent_clusters) == 0:
            self.cluster_intents(self.get_intent_embeddings())

        if self.intent_clusters is not None and len(self.intent_clusters) > 0:
            ai_index = self.kmeans.predict(intent_embedding)[0]
        else:
            ai_index = 0

        chosen_ai = list(self.specialized_ais.values())[ai_index]

        result = await chosen_ai.process(user_input)
        final_result = await self.central_ai.analyze_and_modify(result, user_input)

        self.intent_history[ai_index].append(intent_embedding[0])

        return final_result

    def set_model_manager(self, model_manager):
        self.model_manager = model_manager

    def get_intent_embeddings(self):
        all_intents = []
        for ai_intents in self.intent_history.values():
            all_intents.extend(ai_intents)
        return np.array(all_intents)