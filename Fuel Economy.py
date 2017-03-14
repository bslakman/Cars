
# coding: utf-8

# In[1]:

import pandas as pd
import pandas_profiling


# In[9]:

df = pd.read_csv('vehicles.csv', low_memory=False, index_col='id')


# In[7]:

pandas_profiling.ProfileReport(df)


# In[14]:

make_model = df.groupby(['make', 'model', 'year']).agg('mean')
make_model.head(10)


# In[25]:

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
plt.figure(figsize=(10,6))
make_model[make_model['UCity'] > 0].sort_values(by='UCity', ascending=True)[:25]['UCity'].plot(kind='bar')


# In[ ]:



