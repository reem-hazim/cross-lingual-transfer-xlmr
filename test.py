from transformers import XLMRobertaTokenizer, TrainingArguments, Trainer, XLMRobertaForSequenceClassification
from datasets import load_dataset, load_metric
import pandas as pd
from data_utils import extract_labels

task = "cola"
batch_size = 64

dataset = load_dataset("glue", task)
metric = load_metric('glue', task)

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# dataset = pd.read_csv("clean_CLAMS/ger_data.csv", sep=";")
# dataset["sentence"] = dataset["sentence"].astype(str)

model = XLMRobertaForSequenceClassification('xlm-roberta-base')

args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="matthews_correlation",
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train();


trainer.evaluate();

# labels = extract_labels(dataset.head())

# list1 = [1, 2, 3]
# list2 = [4, 5, 6]

# my_df = pd.DataFrame.from_dict({'list1': list1, 'list2': list2})
# print(my_df)