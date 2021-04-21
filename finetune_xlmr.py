"""Finetune XLMR on CLAMS

Example usage:
    python finetune_xlmr clean_CLAMS
"""
import argparse
import data_utils
import finetuning_utils
import json
import pandas as pd

from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch

from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer
from transformers import TrainingArguments, Trainer
from clams_dataset import CLAMS_Dataset

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning the XLMR model on the the English CLAMS evaluation dataset."
)

parser.add_argument(
    "data_dir",
    type=str,
    help="Folder containing the CLAMS dataset (English, Hebrew and German).",
)

args = parser.parse_args()

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
	weight_decay=0.01,
	learning_rate= 1e-5,
	evaluation_strategy = "epoch",
	load_best_model_at_end=True,
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

trainer.save_model("/scratch/rh3015/MLLU_experiment/finetuned_xlmr");

predictions, label_ids, metrics = trainer.predict(test_data)

print(metrics)

test_preds = pd.DataFrame.from_dict({
	"label": label_ids,
	"pred": predictions.argmax(-1)
	})

test_preds.to_csv("test_predictions.csv", index=False)

# my_hp_space = {"learning_rate": tune.uniform(1e-5, 5e-5),
# 			  "num_train_epochs": tune.choice(range(1, 6)),
# 			  "": tune.choice([4, 8, 16]),
# 			  "gradient_accumulation_steps": tune.choice([1, 2])}

# def compute_objective(metrics):
# 	eval_loss = metrics.pop("eval_loss", None)
# 	return eval_loss

# best_run = trainer.hyperparameter_search(
# 	compute_objective=compute_objective,
# 	direction="minimize",
# 	backend="ray",
# 	n_trials = 7,
# 	hp_space= lambda _:my_hp_space)


# print("Run ID: ", best_run.run_id)
# print("Objective: ", best_run.objective)
# print("Hyperparameters: ", best_run.hyperparameters)