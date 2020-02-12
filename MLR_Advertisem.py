#!/usr/bin/env python
# coding: utf-8

# In[98]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


# In[99]:


data=pd.read_csv('D:DS_TriS/Advertising.csv')
data.head()


# In[121]:


data1=data.drop(['Unnamed: 0'],1)
data1.head()


# In[122]:


corr=data1.corr()
corr.nlargest(4,['sales'])['sales']


# In[123]:


X=data1.drop(['sales'],1)
X.head()


# In[124]:


from sklearn.preprocessing import StandardScaler


# In[125]:


sc=StandardScaler()
X2=sc.fit_transform(X)
X2


# In[126]:


y=data1['sales'].values
y


# In[127]:


from sklearn.model_selection import train_test_split


# In[128]:


X_train,X_test,y_train,y_test=train_test_split(X2,y,test_size=0.3,random_state=1)


# In[129]:


from sklearn.linear_model import LinearRegression


# In[130]:


model=LinearRegression(normalize=True)
model.fit(X_train,y_train)


# In[131]:


model.coef_


# In[132]:


model.intercept_


# In[133]:


y_pred=model.predict(X_test)
y_pred


# In[134]:


y_test


# In[135]:


a=pd.DataFrame({'actual':y_test,'predicted':y_pred})
a.head(10)


# In[136]:


from sklearn.metrics import r2_score


# In[137]:


r2_score(y_test,y_pred)


# In[138]:


model.score(X_train,y_train)


# In[142]:


a=data1.iloc[100:105,:-1]
a


# In[143]:


a=sc.fit_transform(a)
a


# In[144]:


model.predict(a)


# In[ ]:





# In[ ]:




