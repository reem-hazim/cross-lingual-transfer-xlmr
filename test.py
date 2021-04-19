from transformers import XLMRobertaTokenizer
import pandas as pd
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

dataset = pd.read_csv("clean_CLAMS/eng_test.csv")
dataset["label"] = dataset["label"].astype(bool)
dataset["sentence"] = dataset["sentence"].astype(str)

inputs = tokenizer(dataset["sentence"].values.tolist(), 
                      padding=True,
                      truncation= True, 
                      return_tensors='pt', 
                      return_token_type_ids= True,
                      return_attention_mask=True)

print(inputs["input_ids"][0])
print(inputs["attention_mask"][0])