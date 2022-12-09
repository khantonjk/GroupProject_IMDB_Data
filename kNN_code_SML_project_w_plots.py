#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
    Creator: AK
    Use: Project statistical machine learning course Uppsala University
    Date: 2022.12.09

"""


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import usefulfuncs as uff

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png')
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
#plt.style.use('seaborn-white')


# In[3]:


# read data
data = pd.read_csv("train.csv")
#data.head()
data


# In[4]:


# Get headers / categories
cols = list(data.columns)
#len(cols)


# In[5]:


#pd.plotting.scatter_matrix(data.iloc[:,2:4])
#data.iloc[:,1] 
#data['Total words'].tolist()
#data.iloc[1,:].tolist()


# In[6]:


gender_list = data.iloc[:,-1]
cntr = 0
for m in gender_list:
    if m == "Male":
        cntr += 1
print(cntr)
print("% if only guess male: ", cntr/len(gender_list))


# In[7]:


train,test = uff.train2tt(data,0.2)
len(train)


# In[8]:


#N = max len(train) (=total amount of examples training data) 
#, d = max 13 (=total amount of catagories)

def kNN_whatN(k,d,offset,train,test): 
    
    if d > 13:
        print('The dimensions, d, must stay below 13')
        return None
    
    model = skl_nb.KNeighborsClassifier(n_neighbors=k)
    
    X_lst = []
    for i in range(d):
        
        i = i + offset
        if i >= 13:
            i = i-13   
        #print(i)
        
        X_lst.append(cols[i])
        #print(X_lst)

    X_train = train[X_lst]
    Y_train = train[cols[-1]]

    X_test = test[X_lst]
    Y_test = test[cols[-1]]

    model.fit(X_train, Y_train)

    # cell 2
    
    prediction = model.predict(X_test)
    accuary = np.mean(prediction== Y_test)
    return accuary
#pre = kNN_whatN(2,5)
len(train)


# In[9]:


def kNN_with_dimensions(d,max_k,offset,train,test):
    
    #between what k do we operate?
    #first k
    frm_k = 1
    #last k
    too_k = max_k
    # Dimensions
    # +1 to acctualy run to k=too_k
    AccuracyA = np.empty(too_k-frm_k+1)
    kA = np.empty(too_k-frm_k+1)
    
    for k in range(frm_k,too_k+1):
        
        kA[k-1] = k
        Accy = kNN_whatN(k,d,offset,train,test)
        AccuracyA[k-1] = Accy
        
    return kA,AccuracyA 

#kA, AccuracyA = kNN_with_dimensions(1)


# In[10]:


# 0.7555341674687199 only male guess <-- the number to beat
ofst=2 #start catergory_offset
plt_Array=[]
for d in range(1,13+1):
    kA, AccuracyA = kNN_with_dimensions(d,30,ofst,train,test)     
    
    plt_Array.append(AccuracyA)
    
#print(len(plt_Array))
    


# In[14]:


#colorV = ['']
a = 1/14

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for d in range(0,13):
    a = a + 1/14 
    plt.plot(kA,plt_Array[d], label='d='+str(d+1), color='red',alpha=a)

plt.plot(kA, 0.755534*np.ones([len(kA),1]), linestyle='dotted',         label='Only male predicted',color='black')
plt.title('K-NN classification performance in ' + "different" + ' dimensions')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
ax.grid(alpha=0.4)
    #print('k ',kA)
    #print("Acc: ",AccuracyA)


# In[15]:


kNN_whatN(5,12,2,train,test)


# In[16]:


n_runs = 10
K = np.arange(5,6)
misclassification = np.zeros((n_runs, len(K)))
accuracyv2 = np.zeros((n_runs, len(K)))

cls = [0,2,3,4,5,6,7,8,9,10,11,12]
#cls = [3,4,5,7,11]

for i in range(n_runs):
    X_train, X_val, y_train, y_val = skl_ms.train_test_split(        data.iloc[:,cls], data.iloc[:,-1], test_size=0.3)
    for j, k in enumerate(K):
        model = skl_nb.KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        accuracyv2[i, j] = (np.mean(prediction == y_val))
        misclassification[i, j] = (np.mean(prediction != y_val))

#average_mis = np.mean(misclassification, axis=0)
#plt.plot(K, average_mis)
#plt.boxplot(misclassification)
plt.boxplot(accuracyv2)
#plt.title('cross validation error for kNN')
#plt.xlabel('k')
#plt.ylabel('validation error')
plt.show()
# 


# In[17]:


## Now when we have optimized our k-NN 
#we can run it on the actual test data 
#and it will return it's predictions


# In[18]:


# This is the 


# In[19]:


#N = max len(train) (=total amount of examples training data) 
#, d = max 13 (=total amount of catagories)

def kNN_predict(k,d,offset,train,test): 
    
    if d > 13:
        print('The dimensions, d, must stay below 13')
        return None
    
    model = skl_nb.KNeighborsClassifier(n_neighbors=k)
    
    X_lst = []
    for i in range(d):
        
        i = i + offset
        if i >= 13:
            i = i-13   
        
        X_lst.append(cols[i])

    X_train = train[X_lst]
    Y_train = train[cols[-1]]

    X_test = test[X_lst]

    model.fit(X_train, Y_train)
    
    prediction = model.predict(X_test)
    #accuary = np.mean(prediction== Y_test)
    return prediction


# In[20]:


data_test = pd.read_csv("test.csv")
data_train = pd.read_csv("train.csv")
#data_test.iloc[16]


# In[21]:


#input_test = data_test.iloc[:]
#print(pred)


# In[22]:


# Tuning in "k" and "dim" and "offset" to get best result
pred = kNN_predict(5,12,2,data_train,data_test)
#We can only assume this is correct
counter_male = 0
for x in pred:
    if x=='Male':
        counter_male += 1
        
print(counter_male/len(pred))

#Nästa är att..? Skriva i rapporten


# In[23]:


print(pred)


# In[ ]:




