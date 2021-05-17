import pandas as pd
import os 

for filename in os.listdir("../../CLAMS/French"):
	if filename != ".DS_Store":
		df = pd.read_csv(f"../../CLAMS/French/{filename}", sep="\t", names=["label", "sentence"])
		new_df = df.sample(frac=1)
		new_df.to_csv(f"./{filename}", index=False)
