import time

import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import json

# Load pre-trained BERT model and tokenizer
model_name = './bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Function to encode texts in batches using CUDA
def encode_texts_in_batches(texts, batch_size=32):
    all_cls_vectors = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc="编码进度") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            cls_vectors = last_hidden_states[:, 0, :].to('cpu')
            all_cls_vectors.append(cls_vectors)
            pbar.update(1)
    final_cls_vectors = torch.cat(all_cls_vectors, dim=0)
    
    # 将张量转换为DataFrame
    feature_columns = [f'feature_{i}' for i in range(768)]
    return pd.DataFrame(final_cls_vectors.numpy(), columns=feature_columns)
