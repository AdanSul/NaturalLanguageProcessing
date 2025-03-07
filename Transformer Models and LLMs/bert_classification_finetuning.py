import sys
import os
import torch
from datasets import load_from_disk
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from evaluate import load
from datasets import load_dataset


if len(sys.argv) < 2:
    raise ValueError("Usage: python bert_classification_finetuning.py <path/to/imdb_subset>")
    sys.exit(1)


imdb_subset_path = sys.argv[1]


os.environ["WANDB_DISABLED"] = "true"

SEED = 42
torch.manual_seed(SEED)

# subset = load_from_disk(imdb_subset_path)

try:
    subset = load_from_disk(imdb_subset_path)
except Exception:
    dataset = load_dataset("imdb")
    subset = dataset["train"].shuffle(seed=42).select(range(500))


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


tokenized_subset = subset.map(tokenize_function, batched=True)


tokenized_subset = tokenized_subset.rename_column("label", "labels")


train_test_split = tokenized_subset.train_test_split(test_size=0.2, seed=SEED)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    report_to=[]
)


accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test Accuracy: {test_results['eval_accuracy']:.2f}")

