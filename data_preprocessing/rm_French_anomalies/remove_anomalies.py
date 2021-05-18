import pandas as pd
import os 

anomalies = ["retourne", "déménage", "parle"]
plural = ["retournent", "déménagent", "parlent"]
index_to_rm = []

for filename in os.listdir("../../CLAMS/French"):
	if filename != ".DS_Store":
		print(filename)
		df = pd.read_csv(f"../../CLAMS/French/{filename}", sep="\t", names=["label", "sentence"])
		for i in range(df.shape[0]):
			example = df.at[i, "sentence"]
			for j in range(len(anomalies)):
				if anomalies[j] in example and plural[j] not in example and i not in index_to_rm:
					index_to_rm.append(i)
		new_df = df.copy()
		new_df = new_df.drop(index_to_rm)
		index_to_rm.clear()
		new_df.to_csv(f"./new_french_evalsets/{filename}", index=False)
