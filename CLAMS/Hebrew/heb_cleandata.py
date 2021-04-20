import pandas as pandas
import os



#Hebrew dataset
files = ['obj_rel_within_anim.txt', 'subj_rel.txt', 'long_vp_coord.txt',
'prep_anim.txt',              'vp_coord.txt',
'obj_rel_across_anim.txt'   ,   'simple_agrmt.txt']


for file in files:
    f= pd.read_csv(file,header=None)
    f.to_csv('heb_eval.csv', sep='\t',mode='a')


df1=pd.read_csv('heb_eval.csv')
df1.columns=['Sentence']
df1[['num','labels','sentence']] = df1.Sentence.str.strip('"').str.split("\t",expand=True)
df1['label'] = df1.labels.str.strip('"')
df1.drop(['Sentence', 'num','labels'], axis = 1,inplace=True)
df=df1[["label", "sentence"]]
df=df.sample(frac=1)
df=df.drop_duplicates()
df.to_csv("heb_data.csv",index=False)
