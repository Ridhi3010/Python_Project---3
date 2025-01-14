#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.ensemble import RandomForestRegressor 


# In[4]:


from sklearn import metrics 


# In[6]:


gold = pd.read_csv("C:\\Users\\HP\\Downloads\\archive (3)\\gld_price_data.csv")
gold


# In[7]:


#print first 5 row 

gold.head()


# In[8]:


#print last 5 row 

gold.tail()


# In[10]:


#no of rows and columns 

gold.shape


# In[11]:


#getting basic info about data 
gold.info()


# In[12]:


gold.isnull().sum()


# In[13]:


#statistical measure of data 

gold.describe()


# In[ ]:


#correlation 

#postive and negative correlation 


# In[14]:


correlation = gold.corr()


# In[15]:


correlation


# In[19]:


#construct heat map to understand the correlation 

plt.figure(figsize =(8,8))
sns.heatmap(correlation, cbar= True, square = True, fmt='.1f',annot= True, annot_kws={'size':8},cmap='Blues')


# In[20]:


#correlatin values of GLD 


print(correlation['GLD'])


# In[21]:


#check the distribution of the gold price 

sns.distplot(gold['GLD'], color='green')


# In[ ]:


#splitting the features and target 


# In[22]:


x = gold.drop(['Date','GLD'],axis =1)
y = gold['GLD']


# In[23]:


print(x)


# In[24]:


print(y)


# In[25]:


#splitting into train and test data 

x_train,x_test,y_train,y_test =train_test_split(x,y, test_size=0.2, random_state = 2)


# In[26]:


#randomforestregressor 

regressor = RandomForestRegressor(n_estimators=100)


# In[27]:


regressor.fit(x_train,y_train)


# In[ ]:


#model evalutation


# In[28]:


#prediction on test data

test_data_prediction = regressor.predict(x_test)


# In[29]:


print(test_data_prediction)


# In[30]:


#r square error 

error_score = metrics.r2_score(y_test, test_data_prediction)
print("R squared error:", error_score)


# In[ ]:


# compare actual value and predicted value 


# In[31]:


y_test =list(y_test)


# In[32]:



plt.plot(y_test, color ='blue', label = 'Actual values' )
plt.plot(test_data_prediction, color='green', label='Predicted Values')
plt.title('Actual Values vs Predicted Price')
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[ ]:




