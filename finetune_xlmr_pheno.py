"""Finetune XLMR on CLAMS
Example usage:
    python finetune_xlmr clean_CLAMS
"""
import argparse
import data_utils
import finetuning_utils
import pandas as pd

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

train_df = pd.read_csv(f"{args.data_dir}/by_pheno_eng/eng_train.csv")
train_df["label"] = [int(label == "1") for label in train_df["label"]]

val_df = pd.read_csv(f"{args.data_dir}/by_pheno_eng/eng_val.csv")
val_df["label"] = [int(label == "1") for label in val_df["label"]]

test_df = pd.read_csv(f"{args.data_dir}/by_pheno_eng/eng_test.csv")
test_df["label"] = [int(label == "1") for label in test_df["label"]]

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
train_data = CLAMS_Dataset(train_df, tokenizer)
val_data = CLAMS_Dataset(val_df, tokenizer)
test_data = CLAMS_Dataset(test_df, tokenizer)

training_args = TrainingArguments(
	output_dir="xlmr_checkpoints",
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

trainer.save_model("models/finetuned_xlmr_clams_pheno")
