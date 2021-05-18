import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split

# Concatenate datasets for our phenomnena
concat_df = pd.DataFrame(columns=["label", "sentence", "phenomena"])
for filename in os.listdir("../../CLAMS/English"):
	if filename not in [".gitkeep", ".DS_Store"]:
		df = pd.read_csv(os.path.join("../../CLAMS/English", filename), sep="\t", names=["label", "sentence"])
		df["label"] = [int(label == True) for label in df["label"]]
		phen = filename.split(".")[0]
		df["phenomena"] = phen
		concat_df = pd.concat([concat_df, df], ignore_index=True)

print("Concat df size:" + str(concat_df.shape[0]))

# Choose random one out of minimal pair
clean_dataset = pd.DataFrame(columns=concat_df.columns)
for i in range(0, concat_df.shape[0], 2):
	chosen_idx = random.choice([i, i+1])
	clean_dataset = clean_dataset.append(concat_df.iloc[chosen_idx], ignore_index=True)

print("Clean dataset size " + str(clean_dataset.shape[0]))
# Shuffle dataset and drop duplicates
clean_dataset = clean_dataset.sample(frac = 1)
clean_dataset = clean_dataset.drop_duplicates(subset=["sentence"], ignore_index = True)

# Train-val-test split
train_df, test_and_val_df = train_test_split(clean_dataset, test_size = 0.4)
test_df, val_df = train_test_split(test_and_val_df, test_size = 0.5)

train_df.to_csv("eng_train.csv", index=False)
val_df.to_csv("eng_val.csv", index=False)
test_df.to_csv("eng_test.csv", index=False)

# Divide test data by phenomenon
phenomena = ["long_vp_coord.txt", "obj_rel_across_anim.txt", "obj_rel_within_anim.txt", "prep_anim.txt","simple_agrmt.txt", "subj_rel.txt", "vp_coord.txt"]
for filename in phenomena:
    phen = filename.split(".")[0]
    test_phen = test_df[test_df["phenomena"] == phen]
    test_phen = test_phen.drop(axis=1, labels=["phenomena"])
    test_phen.to_csv(f"./test_sets/{phen}.csv", index=False)

