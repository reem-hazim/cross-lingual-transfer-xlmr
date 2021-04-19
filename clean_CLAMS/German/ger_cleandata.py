import pandas as pandas
import os



#German dataset
files = ['obj_rel_within_anim.txt','obj_rel_across_anim.txt','simple_agrmt.txt','subj_rel.txt','long_vp_coord.txt'
        ,'prep_anim.txt','vp_coord.txt']



for file in files:
    f= pd.read_csv(file, header=None, sep=';')
    f.to_csv('ger_eval.csv', sep=';',mode='a')
df1=pd.read_csv('ger_eval.csv',sep=';')
df1.columns=['ind','Sentence']
df1[['labels','sentence']] = df1.Sentence.str.split("\t",expand=True)
df1.drop(['Sentence', 'ind'], axis = 1,inplace=True)
df1.columns=['label','sentence']
df=df1[["label", "sentence"]]
df=df.sample(frac=1)
df1.to_csv("ger_data.csv",sep=';',index=False)
