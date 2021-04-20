from transformers import XLMRobertaTokenizer
import pandas as pd
from data_utils import extract_labels
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

dataset = pd.read_csv("clean_CLAMS/ger_data.csv", sep=";")
dataset["sentence"] = dataset["sentence"].astype(str)
print(dataset.head())
inputs = tokenizer(dataset["sentence"].values.tolist(), 
                      padding=True,
                      truncation= True, 
                      return_tensors='pt', 
                      return_token_type_ids= True,
                      return_attention_mask=True)

print(inputs["input_ids"][0])
print(inputs["attention_mask"][0])
labels = extract_labels(dataset.head())
print(labels)