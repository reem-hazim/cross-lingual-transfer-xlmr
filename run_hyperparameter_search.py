"""Run a hyperparameter search on a RoBERTa model fine-tuned on BoolQ.

Example usage:
    python run_hyperparameter_search.py BoolQ/
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

# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.

train_df = pd.read_csv(f"{args.data_dir}/eng_train.csv")
val_df = pd.read_csv(f"{args.data_dir}/eng_val.csv")
test_df = pd.read_csv(f"{args.data_dir}/eng_test.csv")

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
train_data = CLAMS_Dataset(train_df, tokenizer)

## TODO: Initialize a transformers.TrainingArguments object here for use in
## training and tuning the model. Consult the assignment handout for some
## sample hyperparameter values.
training_args = TrainingArguments(
	output_dir="/scratch/rh3015/",
	num_train_epochs=3,
	per_gpu_train_batch_size=8,
	learning_rate= 1e-5,
	evaluation_strategy = "epoch"
)
## TODO: Initialize a transformers.Trainer object and run a Bayesian
## hyperparameter search for at least 5 trials (but not too many) on the 
## learning rate. Hint: use the model_init() and
## compute_metrics() methods from finetuning_utils.py as arguments to
## trainer.hyperparameter_search(). Use the hp_space parameter to specify
## your hyperparameter search space. (Note that this parameter takes a function
## as its value.)
## Also print out the run ID, objective value,
## and hyperparameters of your best run.

trainer = Trainer(
	model_init = finetuning_utils.model_init,
	compute_metrics= finetuning_utils.compute_metrics,
	args = training_args,
	train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
 )

hyperSpace = {"learning_rate": tune.uniform(1e-5, 5e-5)}

def compute_objective(metrics):
	return metrics.pop("eval_loss", None)

best_run = trainer.hyperparameter_search(
	compute_objective=compute_objective,
	backend="ray",
	n_trials = 7,
	hp_space = lambda _: hyperSpace,
	search_alg = BayesOptSearch(mode="min"))

print("Run ID: ", best_run.run_id)
print("Objective: ", best_run.objective)
print("Hyperparameters: ", best_run.hyperparameters)
