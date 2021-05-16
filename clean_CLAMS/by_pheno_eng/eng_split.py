import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from clams_dataset import CLAMS_Dataset

parser = argparse.ArgumentParser(
    description="Split CLAMs Dataset"
)

parser.add_argument(
    "data_dir",
    type=str,
    help="Folder containing the CLAMS datasets for English.",
)

args = parser.parse_args()


phenomena = ['long_vp_coord.txt','obj_rel_across_anim.txt','obj_rel_within_anim.txt','prep_anim.txt','simple_agrmt.txt','subj_rel.txt','vp_coord.txt']
for filename in os.listdir(args.data_dir):
    if filename in phenomena:
        df = pd.read_csv(os.path.join(args.data_dir, filename), sep="\t", names=["label", "sentence"])
        df["label"] = [int(label == True) for label in df["label"]]
        phenomenon = filename.split(".")[0]
        df["phenomena"] = phenomenon
        df.to_csv(f"./eng_data.csv", index=False,mode='a')

eng_data = pd.read_csv("eng_data.csv",sep=",")
train, validate, test =np.split(eng_data.sample(frac=1, random_state=42), 
                       [int(.6*len(eng_data)), int(.8*len(eng_data))])
train.to_csv("eng_train.csv",index=False)
validate.to_csv("eng_val.csv",index=False)
test.to_csv("eng.test.csv",index=False)