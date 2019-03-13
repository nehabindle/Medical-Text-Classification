
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import nltk
import spacy
from tqdm import tqdm
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


#Dataframe to read the training and mark the labels for Class and Description

Description = []
Class = []

# Training Data
with open('train.dat', 'r') as tr:
    for lines in tr:
        Class.append(lines[0:1])
        Description.append(lines[1:])
trainData = {'Class': Class, 'Description': Description}
df_X = pd.DataFrame(trainData)

#Dataframe to read the testing and mark the labels for Class and Description

Description = []
Class = []

with open('test.dat', 'r') as td:
    for lines in td:
        Description.append(lines[:])
testData = {'Description': Description}
df_y = pd.DataFrame(testData)


# In[47]:


df_X.head()


# In[48]:


#Length of Dataset
print(len(df_X), len(df_y))


# In[49]:


# check class distribution
df_X['Class'].value_counts()


# In[50]:


Y = df_X['Class']


# In[51]:


clean_1 = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
clean_2 = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

# function to clean text data
def clean_desc(desc):
    desc = [clean_1.sub("", line.lower()) for line in desc]
    desc = [clean_2.sub(" ", line) for line in desc]
    return desc


# In[52]:


df_X['Description'] = clean_desc(df_X['Description'])
df_y['Description'] = clean_desc(df_y['Description'])


# In[53]:


from nltk import FreqDist


# In[54]:


# function to plot top n most frequent words
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()
  
  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
  
  # selecting top n most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()


# In[55]:


freq_words(df_X['Description'])


# In[56]:


nlp = spacy.load('en_core_web_sm')


# In[57]:


# tokenization using spaCy
def tokenization(x):
    desc_tokens = []
    for i in tqdm(x):
        i = nlp(i)
        temp = []
        for j in i:
            temp.append(j.text)
        desc_tokens.append(temp)
    
    return desc_tokens


# In[58]:


#Tokenization for Train data
df_X['tokenized_desc'] = tokenization(df_X['Description'])

#Tokenization for Test Data
df_y['tokenized_desc'] = tokenization(df_y['Description'])


# In[59]:


# function to remove stopwords
def remove_stopwords(desc):
    s = []
    for r in tqdm(desc):
        s_2 = []
        for token in r:
            if nlp.vocab[token].is_stop == True:
                continue
            else:
                s_2.append(token)
        s.append(" ".join(s_2))    
        
    return s


# In[60]:


df_X['Desc_cleaned'] = remove_stopwords(df_X['tokenized_desc'])


# In[61]:


freq_words(df_X['Desc_cleaned'])


# In[62]:


# remove anything smaller then length 4 in training set
df_X['Desc_cleaned'] = df_X['Desc_cleaned'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>4]))


# In[63]:


freq_words(df_X['Desc_cleaned'])


# In[64]:



df_y['Desc_cleaned'] = remove_stopwords(df_y['tokenized_desc'])


# In[65]:


# remove anything smaller then length 4 in test set
df_y['Desc_cleaned'] = df_y['Desc_cleaned'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>4]))


# In[66]:


df_X['Desc_cleaned'][0]


# In[67]:


#from sklearn.model_selection import train_test_split


# In[68]:


#train_x, train_y, test_x, test_y = train_test_split(df_X['Desc_cleaned'],df_X['Class'], test_size=0.3, random_state=42)


# In[69]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[70]:


# build TF-IDF features for train data
cv = TfidfVectorizer(use_idf=True, min_df=3, max_df=0.5, ngram_range=(1,2),
                        sublinear_tf=True,max_features=5000)
cv_train = cv.fit_transform(df_X['Desc_cleaned'])


# In[71]:


cv_train


# In[72]:


#build TF-IDF features for test data
cv_test = cv.transform(df_y['Desc_cleaned'])
cv_test


# In[73]:


#cv_train_x = cv.fit_transform(train_x)


# In[74]:


#cv_test_x =cv.transform(train_y) 


# In[78]:


import math
import operator


# In[79]:


def cosine_similarity(mat2,mat1):
    cosine_mat = mat1.dot(mat2.T)
    return cosine_mat


# In[80]:


from collections import defaultdict
def KNN_CS(train_matrix, test_matrix, labels, kn=4,eps=0.3):
    cos_sim = cosine_similarity(train_matrix,test_matrix)
    print(cos_sim.shape)
    temp = []
    length=len(labels)
    indices = np.arange(0,length)
    for cal in cos_sim:
        temp.append(list(zip(indices,cal.data)))
    sortedVals=[]
    for i in temp:
        i.sort(key=lambda a:a[1], reverse=True)
        sortedVals.append(i)
        
    selectedVals=[]
    for temp in sortedVals:
        builtin=[]
        for i,val in enumerate(temp):
            if(val[1] > eps):
                builtin.append(val)
            #if(i>=k-1):
                #if(len(inside)==0):
                    #inside.append(sim[0])
                #break;
            if(i>= kn-1) and (len(selectedVals)==0):
                builtin.append(temp[0])
                break
        selectedVals.append(builtin)
    
    
    print(len(selectedVals))
    selected=[]
    #print(finalList[0])
    for s in selectedVals:
        #print(q)
        dip = defaultdict(float)
        for i in s:
            dip[i[0]]+= i[1]
        selected.append(dip)
    print(len(dip))    
    print(len(selected))
    selected = [sorted(dip.items(),key=lambda x:x[1],reverse=True)[0][0] for dip in selected]
    
    pred_results=[labels[i] for i in selected]
    return pred_results


# In[86]:


# Actual data predictions
spaces = []
spaces = KNN_CS(cv_train, cv_test, Y,4,0.03)   


# In[87]:


spaces


# In[105]:


pred = []
for sublist in spaces:
    for item in sublist:
        pred.append(item)
    


# In[106]:


output_file = open('predictions.dat', 'w')
for i in pred:
     output_file.write(i + '\n')
output_file.close()

print(len(pred))


# In[101]:


# spaces1 = []
# spaces1 = KNN_CS(cv_train_x, cv_test_x, test_x,4,0.03)   


# In[102]:


# predy = []

# for sublist in spaces1:
#     for item in sublist:
#         predy.append(item)


# In[103]:


# from sklearn import metrics


# In[104]:


# print("Accuracy:",metrics.accuracy_score(test_y, pred))

