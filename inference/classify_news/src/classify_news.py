import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from datasets import Dataset
from transformers import BertTokenizerFast, AutoModelForSequenceClassification
from typing import List

def preprocess(new_data: List, tokenizer):
    encoded_text = tokenizer(
        new_data,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=False,
        padding='max_length',
        max_length=32, 
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']
    
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=32)
    
    return dataloader

def infer(dataloader, model, device):
    model.eval()
    
    predictions = []
    
    for batch in dataloader:
        batch = tuple(bt.to(device) for bt in batch)
        
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1]
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs["logits"].cpu().detach().numpy()
        prediction = np.argmax(logits, axis=1).flatten()
        predictions.extend(prediction)
    
    return predictions
