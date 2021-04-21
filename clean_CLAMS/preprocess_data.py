import pandas as pd 

train_df = pd.read_csv("eng_train.csv")
test_df = pd.read_csv("eng_test.csv")
val_df = pd.read_csv("eng_val.csv")

test_df["label"] = [int(label == True) for label in test_df["label"]]
val_df["label"] = [int(label == True) for label in val_df["label"]]
train_df["label"] = [int(label == "True") for label in train_df["label"]]

test_df.to_csv("eng_test.csv", index=False)
train_df.to_csv("eng_train.csv", index=False)
val_df.to_csv("eng_val.csv", index=False)