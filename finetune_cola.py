import data_utils
import finetuning_utils
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef

# from ray import tune
# from ray.tune.suggest.bayesopt import BayesOptSearch

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric

task = "cola"
model_checkpoint = "xlm-roberta-base"
metric_name="matthews_correlation"
batch_size = 16

dataset = load_dataset("glue", task)

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

model = XLMRobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

args = TrainingArguments(
    "cola_checkpoints",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    metrics = {}
    # metrics["precision"], metrics["recall"], metrics["f1"], _ = precision_recall_fscore_support(labels, preds, pos_label=1, average="binary")
    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["mathews_correlation"] = matthews_corrcoef(labels, preds)
    return metrics


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("models/finetuned_xlmr_cola");

results = trainer.evaluate()

print("Evaluation results:")
print(results)

# print(metrics)

# test_preds = pd.DataFrame.from_dict({
#     "label": label_ids,
#     "pred": predictions.argmax(-1)
#     })


# test_preds.to_csv("results/preds_cola.csv", index=False)


