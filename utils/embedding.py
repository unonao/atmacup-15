import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm.auto import tqdm


class TextDataset(Dataset):
    def __init__(self, texts, model_name="intfloat/multilingual-e5-small"):
        self.texts = texts
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text, truncation=True, max_length=512, padding="max_length", return_tensors="pt"
        )
        return {"input_ids": inputs["input_ids"].squeeze(), "attention_mask": inputs["attention_mask"].squeeze()}


class TextEmbedder:
    def __init__(self, device=None, model_name="intfloat/multilingual-e5-small"):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)

    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_embeddings(self, texts, batch_size=16):
        dataset = TextDataset(texts, self.model_name)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs = {name: tensor.to(self.device) for name, tensor in batch.items()}
                outputs = self.model(**inputs)
                embeddings.append(self.average_pool(outputs.last_hidden_state, inputs["attention_mask"]).cpu().numpy())
        embeddings = np.concatenate(embeddings)
        return embeddings
