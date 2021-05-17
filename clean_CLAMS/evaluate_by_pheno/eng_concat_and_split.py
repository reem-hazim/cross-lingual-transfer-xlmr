import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

new_df = pd.DataFrame(columns=["label", "sentence", "phenomena"])
for filename in os.listdir("../../CLAMS/English"):
    if filename not in [".gitkeep", "eng_cleandata.py"]:
        df = pd.read_csv(os.path.join("../../CLAMS/English", filename), sep="\t", names=["label", "sentence"])
        df["label"] = [int(label == True) for label in df["label"]]
        phenomenon = filename.split(".")[0]
        df["phenomena"] = phenomenon
        new_df = pd.concat([df, new_df])
        new_df = new_df.drop_duplicates(subset=["sentence"], ignore_index = True)

new_df.to_csv("./eng_full_dataset.csv", index=False,mode='a')

train, validate, test =np.split(new_df.sample(frac=1, random_state=42), 
                       [int(.6*len(new_df)), int(.8*len(new_df))])
train.to_csv("eng_train.csv",index=False)
validate.to_csv("eng_val.csv",index=False)
test.to_csv("eng.test.csv",index=False)

# Split test data
phenomena = ["long_vp_coord.txt", "obj_rel_across_anim.txt", "obj_rel_within_anim.txt", "prep_anim.txt","simple_agrmt.txt", "subj_rel.txt", "vp_coord.txt"]
for filename in phenomena:
	phen = filename.split(".")[0]
	test_phen = test[test["phenomena"] == phen]
	test_phen = test_phen.drop(axis=1, labels=["phenomena"])
	test_phen.to_csv(f"./test_by_phenomenon/{phen}.csv")
