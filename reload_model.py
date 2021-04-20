import argparse
import data_utils
import finetuning_utils
import json
import pandas as pd

from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch

from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer
from transformers import TrainingArguments, Trainer, XLMRobertaForSequenceClassification
from clams_dataset import CLAMS_Dataset

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning the XLMR model on the the English CLAMS evaluation dataset."
)

parser.add_argument(
    "data_dir",
    type=str,
    help="Folder containing the CLAMS dataset (English, Hebrew and German).",
)

parser.add_argument(
    "model_dir",
    type=str,
    help="Folder containing the saved model."
)

args = parser.parse_args()

# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.

train_df = pd.read_csv(f"{args.data_dir}/eng_train.csv")
val_df = pd.read_csv(f"{args.data_dir}/eng_val.csv")
test_df = pd.read_csv(f"{args.data_dir}/eng_test.csv")

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
train_data = CLAMS_Dataset(train_df, tokenizer)
val_data = CLAMS_Dataset(val_df, tokenizer)
test_data = CLAMS_Dataset(test_df, tokenizer)

training_args = TrainingArguments(
	output_dir="/scratch/rh3015/MLLU_experiment",
	num_train_epochs=5,
	per_device_train_batch_size=64,
	per_device_eval_batch_size=64,
	learning_rate= 2e-5,
	evaluation_strategy = "epoch",
	load_best_model_at_end=True,
)

def model_init():
	model = XLMRobertaForSequenceClassification.from_pretrained(args.model_dir)
	return model

trainer = Trainer(
	model_init = model_init,
	compute_metrics= finetuning_utils.compute_metrics,
	args = training_args,
	train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
 )

predictions, label_ids, metrics = trainer.predict(test_data)

print("predictions: ")
print(predictions)

print("label ids: ")
print(label_ids)