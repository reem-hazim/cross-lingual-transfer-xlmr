from transformers import XLMRobertaTokenizer
import pandas as pd
from data_utils import extract_labels
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

dataset = pd.read_csv("clean_CLAMS/ger_data.csv", sep=";")
dataset["sentence"] = dataset["sentence"].astype(str)

inputs = tokenizer(dataset["sentence"].values.tolist(), 
                      padding=True,
                      truncation= True, 
                      return_tensors='pt', 
                      return_token_type_ids= True,
                      return_attention_mask=True)

labels = extract_labels(dataset.head())

list1 = [1, 2, 3]
list2 = [4, 5, 6]

my_df = pd.DataFrame.from_dict({'list1': list1, 'list2': list2})
print(my_df)