
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


# In[28]:

plt.figure(figsize=(10,6))
make_model[make_model['UCity'] > 0].sort_values(by='UCity', ascending=False)[:25]['UCity'].plot(kind='bar', color='red', alpha=0.5)


# In[15]:

import numpy as np


# In[17]:

all_car_info = pd.read_csv('all_car_info.csv')
all_car_info['mileage'] = all_car_info.apply(lambda row: row['mileage'] if row['mileage'] < 1E6 else np.nan, axis=1)
all_car_info['price'] = all_car_info.apply(lambda row: row['price'] if row['price'] < 5E5 else np.nan, axis=1)
fuel_years = df.groupby('year').agg(['mean', 'count'])
craigslist_years = all_car_info.groupby('year').agg(['mean', 'count'])
craigslist_years = craigslist_years[craigslist_years[('mileage', 'count')] >= 5]
print fuel_years.head(10)
print craigslist_years.head(10)


# In[18]:

craigslist_years = craigslist_years.join(fuel_years, how='inner')
craigslist_years.head(10)


# In[ ]:



