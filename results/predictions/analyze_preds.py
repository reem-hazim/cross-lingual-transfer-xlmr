import os
import pandas as pd
from transformers import XLMRobertaTokenizer
import argparse

parser = argparse.ArgumentParser(
    description="Evaluate finetuned XLMR on CLAMS dataset."
)

parser.add_argument(
    "lang",
    type=str,
    help="Language to evaluate",
)

parser.add_argument(
    "phen",
    type=str,
    help="Phenomenon to evaluate",
)

args = parser.parse_args()
lang = args.lang
phen = args.phen

old_df = pd.read_csv(f"../../CLAMS/{lang}/{phen}.txt", sep="\t", names=["label", "sentence"])
pred_df = pd.read_csv(f"./main_results/{lang.lower()}/xlmr_{phen}_{lang.lower()}_preds.csv")

new_df = pd.DataFrame(columns=["label", "pred", "sentence"])
for i in range(pred_df.shape[0]):
	if pred_df.iloc[i]['label'] != pred_df.iloc[i]["pred"]:
		new_df.loc[len(new_df.index)] = [pred_df.at[i, "label"], pred_df.at[i, "pred"], old_df.at[i, "sentence"]]
		print(pred_df.iloc[i], end="\t")
		print(old_df.iloc[i]["sentence"])

new_df.to_csv(f"./analyze_preds/wrong_preds_{lang}_{phen}.csv", index=False)


	