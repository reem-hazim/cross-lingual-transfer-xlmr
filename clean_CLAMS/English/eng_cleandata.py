import pandas as pandas
import os



#English dataset
files = ['long_vp_coord.txt'    ,  'prep_inanim.txt', 'prep_anim.txt',
'npi_across_anim.txt'   ,            'reflexive_sent_comp.txt',
'npi_across_inanim.txt',           'reflexives_across.txt',
'obj_rel_across_anim.txt',           'sent_comp.txt',
'obj_rel_across_inanim.txt',       'simple_agrmt.txt',
'obj_rel_no_comp_across_anim.txt',   'simple_npi_anim.txt',
'obj_rel_no_comp_across_inanim.txt', 'simple_npi_inanim.txt',
'obj_rel_no_comp_within_anim.txt',   'simple_reflexives.txt',
'obj_rel_no_comp_within_inanim.txt', 'subj_rel.txt',
'obj_rel_within_anim.txt',         'vp_coord.txt',
'obj_rel_within_inanim.txt']


for file in files:
    f= pd.read_csv(file,header=None)
    f.to_csv('eng_eval.csv', sep='\t',mode='a')


df1=pd.read_csv('eng_eval.csv')
df1.columns=['Sentence']
df1[['num','labels','sentence']] = df1.Sentence.str.strip('"').str.split("\t",expand=True)
df1['label'] = df1.labels.str.strip('"')
df1.drop(['Sentence', 'num','labels'], axis = 1,inplace=True)
df=df1[["label", "sentence"]]
df=df.sample(frac=1)
df.to_csv("eng_data.csv",index=False)



