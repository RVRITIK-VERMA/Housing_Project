#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


cols=df_train.columns


# In[ ]:


df_train[cols] = scaler.fit_transform(df_train[cols])


# In[ ]:


df_train.describe()


# In[ ]:


plt.figure(figsize=[6,6])
plt.scatter(df_train.price, df_train.sqft_living)
plt.show()

