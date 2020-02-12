#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Wine_Quality_Multiple_linear_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("D:DS_TriS/winequality.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.isnull().any()


# In[61]:


#Divide attributes/features(X) and labels(Y)
X = pd.DataFrame(data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']])
Y = data['quality'].values


# In[63]:


X


# In[64]:


Y


# In[65]:


import seaborn as sn


# In[66]:


plt.figure(figsize = (8,5))
plt.tight_layout()
sn.distplot(data['quality'])


# In[67]:


#spliting data into train test
from sklearn.model_selection import train_test_split


# In[68]:


X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)


# In[69]:


#Tain our model
from sklearn.linear_model import LinearRegression


# In[70]:


regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[71]:


#Coefficients of our model
coeff = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])


# In[72]:


coeff


# In[73]:


y_pred = regressor.predict(X_test)


# In[74]:


y_pred


# In[75]:


df = pd.DataFrame({'Acutal':y_test,'Predicted':y_pred})


# In[88]:


df1 = df.head(30)
df1


# In[90]:


df1.plot(kind='bar',figsize=(15,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='blue')
plt.show()


# In[92]:


from sklearn import metrics


# In[93]:


#Mean_Absolute_Error(MSE)
metrics.mean_absolute_error(y_test,y_pred)


# In[95]:


#Mean_Squared_Error(MSE)
metrics.mean_squared_error(y_test,y_pred)


# In[97]:


#Root_Mean_Squared_Error(MSE)
np.sqrt(metrics.mean_squared_error(y_test,y_pred))


# In[100]:


#RMSE is 0.62 which is slightly greater than 10% of mean value 5.63
np.mean(Y)


# In[ ]:




