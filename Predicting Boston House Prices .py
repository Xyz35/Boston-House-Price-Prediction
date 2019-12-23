#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

boston=load_boston()
print(boston)


# In[3]:


#transform data set into a dataframe
df_x =pd.DataFrame(boston.data,columns=boston.feature_names)
df_y =pd.DataFrame(boston.target)

df_x.head(5)


# In[5]:



df_x.describe()


# In[6]:


#model definition
lm=linear_model.LinearRegression()


# In[14]:


x_train,x_test,y_train,y_test= train_test_split(df_x,df_y,test_size=0.4,random_state=44)


# In[15]:


#train model
lm.fit(x_train,y_train)


# In[16]:


y_pred=lm.predict(x_test)
print(y_pred)


# In[17]:


print(y_test)


# In[19]:


#calculate the accuracy of the model using RMSE
acc=np.mean((y_pred-y_test)**2)
print(np.sqrt(acc))

