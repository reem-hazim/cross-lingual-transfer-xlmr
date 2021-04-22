from sklearn.model_selection import train_test_split
import pandas as pd

eng_df = pd.read_csv('eng_data.csv')
train_df, test_and_val_df = train_test_split(eng_df, test_size = 0.4)
test_df, val_df = train_test_split(test_and_val_df, test_size = 0.5)

test_df["label"] = [int(label == True) for label in test_df["label"]]
val_df["label"] = [int(label == True) for label in val_df["label"]]
train_df["label"] = [int(label == "True") for label in train_df["label"]]

train_df.to_csv("eng_train.csv", index=False)
val_df.to_csv("eng_val.csv", index=False)
test_df.to_csv("eng_test.csv", index=False)
