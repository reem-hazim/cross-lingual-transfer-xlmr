import torch
import random
import pandas as pd
import os

unknown_id = 3

def concat_eng_datasets(folder_name):
  df = pd.DataFrame(columns=["label", "sentence"])
  phenomena = ["long_vp_coord.txt", "obj_rel_across_anim.txt", "obj_rel_within_anim.txt", "prep_anim.txt", "simple_agrmt.txt", "subj_rel.txt", "vp_coord.txt"]
  for phenomenon in phenomena:
    new_df = pd.read_csv(os.path.join(folder_name, phenomenon), sep="\t", names=["label", "sentence"])
    df = pd.concat([df, new_df])
  return df

def choose_from_pair(dataset):
  dataset["sentence"] = dataset["sentence"].astype(str)
  # Choose random one out of minimal pair
  new_dataset = pd.DataFrame(columns=dataset.columns)

  for i in range(0, dataset["sentence"].size, 2):
    chosen_idx = random.choice([i, i+1])
    print(dataset.iloc[chosen_idx])
    new_dataset = new_dataset.append(dataset.iloc[chosen_idx], ignore_index=True)
  return new_dataset

def encode_data(dataset, tokenizer, max_seq_length=128):
    """Featurizes the dataset into input IDs and attention masks for input into a
     transformer-style model.

  Args:
    dataset: A Pandas dataframe containing the data to be encoded.
    tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
      tokenize the data.
    max_seq_length: Maximum sequence length to either pad or truncate every
      input example to.

  Returns:
    input_ids: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing token IDs for the data.
    attention_mask: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing attention masks for the data.
  """
    # tokenize
    inputs = tokenizer(new_dataset["sentence"].values.tolist(), 
                      padding=True,
                      truncation= True, 
                      return_tensors='pt', 
                      return_token_type_ids= True,
                      return_attention_mask=True)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Remove out of vocab examples
    print("Out of vocab examples:")
    rm_idx = []
    
    for i in range(0, new_dataset["sentence"].size, 2):
      if unknown_id in input_ids[i]:
        rm_idx.append(i)
    for i in rm_idx:
      print(new_dataset.at[i, "sentence"])
      input_ids = torch.cat((input_ids[:i,:], input_ids[i+1:,:]))
      attention_mask = torch.cat((attention_mask[:i,:], attention_mask[i+2:,:]))
    
    return input_ids, attention_mask


def extract_labels(dataset):
    """Converts labels into numerical labels.

  Args:
    dataset: A Pandas dataframe containing the labels in the column 'label'.

  Returns:
    labels: A list of integers corresponding to the labels for each example,
      where 0 is False and 1 is True.
  """
    labels = dataset["label"].tolist()
    return labels
