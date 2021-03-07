#!/usr/bin/env python
# coding: utf-8

# Imports

from tqdm.notebook import tqdm
import os
import json
import pandas as pd
import time
from multiprocessing import Pool
import numpy as np
from pandas.core.common import flatten
from sklearn.utils import shuffle

# Define Path for CORD 19 Dataset

path = "document_parses/pdf_json/"

# Function to extract title and abstract from json files.

def process_json(f):
    text_content = '' 
    working_dict = dict({"Content": []})
    content = json.load(open(path+f,'r'))
    if 'metadata' in content and 'title' in content['metadata']:
        text_content = text_content  +' '+ content['metadata']['title']
    if 'abstract' in content:
        for abst in content['abstract'] :
            text_content = text_content+' '+abst['text']
    working_dict['Content']=text_content.lstrip().lower()
    return working_dict

# Run Function via Pool for faster processing

start=time.time()
pool = Pool(20)                                
json_data=pool.map(process_json, os.listdir(path))  
pool.close()
end=time.time()
print(end-start)

# Basic Data Cleaning, Preprocessing

data=pd.DataFrame(json_data)
data=data.drop_duplicates(subset=['Content'])
data['len']=data['Content'].apply(lambda x:len(x.split()))
data=data[data.len>50]
data=shuffle(data,random_state=0)

# Save Training Set for LanguageModel to Text File

documents="\n".join(data['Content'][0:80000])
file = open("data/train.txt","w")
file.write(documents) 
file.close() 

# Save Validation Set for LanguageModel to Text File

documents="\n".join(data['Content'][80000:])
file = open("data/eval.txt","w")
file.write(documents) 
file.close() 