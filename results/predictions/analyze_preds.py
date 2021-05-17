import os
import pandas as pd
from CLAMS_Dataset import CLAMS_Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

old_df = pd.read_csv("../../CLAMS/Russian/simple_agrmt.txt", sep="\t", names=["label", "sentence"])
print(old_df.head())
clams = CLAMS_Dataset(old_df, tokenizer)
print(old_df.at[9, "sentence"])
print(clams[9])
print(old_df.at[8, "sentence"])
print(clams[8])
print("\n")
print(old_df.at[22, "sentence"])
print(clams[22])
print(old_df.at[23, "sentence"])
print(clams[23])
print(tokenizer.decode([0, 7, 2]))
# print(clams[10])

pred_df = pd.read_csv("./russian/xlmr_simple_agrmt_russian_preds.csv")
wrong_ex = pred_df[pred_df["label"] != pred_df["pred"]]
print(wrong_ex.head(20))
print('\n')

# for filename in os.listdir(f'./{lang}'):
# 	if filename != ".DS_Store":
# 		print(filename)
# 		pred_df = pd.read_csv(os.path.join(f'./{lang}', filename))
# 		# wrong_ex = df[df["label"] != df["pred"]]
# 		print(clams[9])
# 		# print(wrong_ex.head(20))
# 		# print('\n')


	