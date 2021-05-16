import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split

# Concatenate datasets for our phenomnena
df = pd.DataFrame(columns=["label", "sentence"])
phenomena = ["long_vp_coord.txt", "obj_rel_across_anim.txt", "obj_rel_within_anim.txt", "prep_anim.txt", "simple_agrmt.txt", "subj_rel.txt", "vp_coord.txt"]
for phenomenon in phenomena:
	new_df = pd.read_csv(os.path.join("CLAMS/English", phenomenon), sep="\t", names=["label", "sentence"])
	df = pd.concat([df, new_df], ignore_index=True)

df["sentence"] = df["sentence"].astype(str)

# Choose random one out of minimal pair
clean_dataset = pd.DataFrame(columns=df.columns)

for i in range(0, df["sentence"].size, 2):
	chosen_idx = random.choice([i, i+1])
	clean_dataset = clean_dataset.append(df.iloc[chosen_idx], ignore_index=True)

# Train-val-test split
train_df, test_and_val_df = train_test_split(clean_dataset, test_size = 0.4)
test_df, val_df = train_test_split(test_and_val_df, test_size = 0.5)

test_df["label"] = [int(label == True) for label in test_df["label"]]
val_df["label"] = [int(label == True) for label in val_df["label"]]
train_df["label"] = [int(label == True) for label in train_df["label"]]

train_df.to_csv("clean_CLAMS/experiment_2/eng_train_2.csv", index=False)
val_df.to_csv("clean_CLAMS/experiment_2/eng_val_2.csv", index=False)
test_df.to_csv("clean_CLAMS/experiment_2/eng_test_2.csv", index=False)
