import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from tqdm.notebook import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)
import json
import re

# 針對是否為新聞語句，預先人工標註10000筆資料


# data format
# segment_data.csv:
#     news: {0, 1}
#     text: str

data = pd.read_csv("data/segment_data.csv", encoding="big5hkscs")
epochs = 5
lr = 2e-5

ind = np.arange(len(data)) < 10000
train = data[ind].sample(frac=1)
test = data[~ind].sample(frac=1)

dataset = DatasetDict({
    'train': Dataset.from_pandas(train),
    'test': Dataset.from_pandas(test),
})

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(
    "ckiplab/albert-tiny-chinese",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)

def preprocess_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", max_length=32, truncation=True
    )
tokenized_dataset = dataset.map(preprocess_function, batched=True)

tokenized_dataset.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "news"]
)
dataloader_train = torch.utils.data.DataLoader(
    tokenized_dataset["train"], batch_size=32
)
dataloader_valid = torch.utils.data.DataLoader(tokenized_dataset["test"], batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def score_fun(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat), accuracy_score(labels_flat, preds_flat)

def evaluate(dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch.values())

        inputs = {
            "input_ids": batch[1],
            "attention_mask": batch[3],
            "labels": batch[0],
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=50, num_training_steps=len(dataloader_train) * epochs
)


for epoch in tqdm(range(1, epochs + 1)):
    model.train()

    loss_train_total = 0

    progress_bar = tqdm(
        dataloader_train, desc="Epoch {:1d}".format(epoch), leave=False, disable=False
    )
    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch.values())

        inputs = {
            "input_ids": batch[1],
            "attention_mask": batch[3],
            "labels": batch[0],
        }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(
            {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
        )

    torch.save(model.state_dict(), f"model/news_classify_epoch_{epoch}.model")

    tqdm.write(f"\nEpoch {epoch}")

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f"Training loss: {loss_train_avg}")

    val_loss, predictions, true_vals = evaluate(dataloader_valid)
    val_f1, val_acc = score_fun(predictions, true_vals)
    tqdm.write(f"Validation loss: {val_loss}")
    tqdm.write(f"F1 score: {val_f1}")
    tqdm.write(f"accuracy: {val_acc}")