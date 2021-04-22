import data_utils
import finetuning_utils
import pandas as pd

# from ray import tune
# from ray.tune.suggest.bayesopt import BayesOptSearch

from transformers import XLMRobertaTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric

task = "cola"
model_checkpoint = "xlm-roberta-base"
metric_name="matthews_correlation"
batch_size = 64

dataset = load_dataset("glue", task)
metric = load_metric('glue', task)

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

model = XLMRobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer = Trainer(
    model_init = finetuning_utils.model_init,
    compute_metrics= finetuning_utils.compute_metrics,
    args = training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
 )

trainer.train()

trainer.save_model("models/finetuned_xlmr_cola");

predictions, label_ids, metrics = trainer.predict(test_data)

print(metrics)

test_preds = pd.DataFrame.from_dict({
    "label": label_ids,
    "pred": predictions.argmax(-1)
    })


test_preds.to_csv("results/preds_cola.csv", index=False)


