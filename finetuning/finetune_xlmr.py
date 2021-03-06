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
from CLAMS_Dataset import CLAMS_Dataset

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning the XLMR model on the the English CLAMS evaluation dataset."
)

parser.add_argument(
    "data_dir",
    type=str,
    help="Folder containing the CLAMS dataset",
)

args = parser.parse_args()

train_df = pd.read_csv(f"../{args.data_dir}/eng_train.csv")
val_df = pd.read_csv(f"../{args.data_dir}/eng_val.csv")
# test_df = pd.read_csv(f"../{args.data_dir}/eng_test.csv")

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
train_data = CLAMS_Dataset(train_df, tokenizer)
val_data = CLAMS_Dataset(val_df, tokenizer)
# test_data = CLAMS_Dataset(test_df, tokenizer)

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

trainer.save_model("../models/finetuned_xlmr_rm_french_anomalies");

# predictions, label_ids, metrics = trainer.predict(test_data)

# print('\n')
# print("Test Results:")
# print(metrics)

# test_preds = pd.DataFrame.from_dict({
# 	"label": label_ids,
# 	"pred": predictions.argmax(-1)
# 	})

# test_preds.to_csv("results/predictions/baseline_xlmr_predictions_2.csv", index=False)
