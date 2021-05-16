"""Evaluate XLMR on CLAMS

Example usage:
    python eval_lang CLAMS/Hebrew
"""
import argparse
import data_utils
import finetuning_utils
import pandas as pd
import os

from transformers import XLMRobertaTokenizer,XLMRobertaForSequenceClassification
from transformers import TrainingArguments, Trainer
from clams_dataset import CLAMS_Dataset

parser = argparse.ArgumentParser(
    description="Evaluate finetuned XLMR on CLAMS dataset."
)

parser.add_argument(
    "lang",
    type=str,
    help="Language to evaluate",
)

parser.add_argument(
    "data_dir",
    type=str,
    help="Folder containing the CLAMS datasets for that language.",
)

args = parser.parse_args()

lang = args.lang.lower()

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

def model_init():
	model = XLMRobertaForSequenceClassification.from_pretrained("models/finetuned_xlmr_clams_pheno")
	return model


for filename in os.listdir(args.data_dir):
	if filename != f"{lang}_cleandata.py" and filename != ".gitkeep":
		test_df = pd.read_csv(os.path.join(args.data_dir, filename), sep="\t", names=["label", "sentence"])
		test_df["label"] = [int(label == True) for label in test_df["label"]]
		test_data = CLAMS_Dataset(test_df, tokenizer)
		trainer = Trainer(model_init = model_init, compute_metrics = finetuning_utils.compute_metrics, tokenizer=tokenizer)
		predictions, label_ids, metrics = trainer.predict(test_data)
		phenomenon = filename.split(".")[0]
		print(phenomenon)
		print(metrics)
		print('\n')
		test_preds = pd.DataFrame.from_dict({ "label": label_ids, "pred": predictions.argmax(-1)})
		test_preds.to_csv(f"./results/predictions/{lang}/xlmr_{phenomenon}_{lang}_preds.csv", index=False)

