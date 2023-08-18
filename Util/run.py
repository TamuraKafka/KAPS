from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from transformers import DataCollatorWithPadding
import torch
import transformers as ppb
import warnings
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

accuracy = evaluate.load("accuracy")
warnings.filterwarnings('ignore')

# 加载数据集
dataset = load_dataset("yelp_review_full")
dataset["train"][100]

# 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "1 star", 1: "2 star", 2: "3 star", 3: "4 star", 4: "5 star"}
label2id = {"1 star": 0, "2 star": 1, "3 star": 2, "4 star": 3, "5 star": 4}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=5, id2label=id2label, label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    push_to_hub_model_id="xuejiubert",
    push_to_hub_organization="xuejiubert",
    push_to_hub_token="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"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()