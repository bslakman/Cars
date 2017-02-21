
# coding: utf-8

# In[22]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set_style('ticks')
import numpy as np


# ## Craigslist Data
# This data has been scraped from boston.craigslist.org using the BeautifulSoup, requests and re modules; see Craiglist.ipynb notebook for code and more details.

# In[16]:

get_ipython().magic(u'store -r all_car_info')
all_car_info.head(3)


# In[9]:

all_car_info.plot.scatter('mileage', 'price')
plt.xlim(0,3E5)
plt.xlabel('Mileage', fontdict={'fontsize': 14})
plt.ylim(0,1E5)
plt.ylabel('Price', fontdict={'fontsize': 14})
plt.title('Car prices vs. Mileage', fontdict={'fontsize': 16})


# In[10]:

all_car_info.plot.scatter('year', 'price')
plt.ylim(0,2E5)
plt.ylabel('Price', fontdict={'fontsize': 14})
plt.xlim(1950,2020)
plt.xlabel('Year', fontdict={'fontsize': 14})
plt.title('Car prices vs. Year', fontdict={'fontsize': 16})


# In[17]:

get_ipython().magic(u'store -r regions')
regions.head(3)


# In[19]:

ax = regions['price','mean'].plot.bar(position=0, width=0.3, alpha=0.8, legend=True)
ax.set_title('Average Price and Mileage of Used Cars in Greater Boston, by region', fontdict={'fontsize':16})
ax.set_xlabel('City/Town', fontdict={'fontsize':14})
ax.set_xticklabels(regions.index, fontdict={'fontsize':12})
ax.set_ylabel('Price($)', fontdict={'fontsize':14})
ax.set_yticklabels(range(0,20000,2500), fontdict={'fontsize':12})
ax = regions['mileage','mean'].plot.bar(secondary_y=True, color='red', position=1, width=0.3, alpha=0.5, legend=True)
ax.set_ylabel('Mileage', fontdict={'fontsize':14})
ax.set_yticklabels(range(0,160000,20000), fontdict={'fontsize':12})
sns.despine(top=True, right=False)
fig=ax.get_figure()
fig.set_size_inches(10,4)


# ## Modeling
# A linear regression model was trained on year and mileage using scikit-learn.

# In[24]:

get_ipython().magic(u'store -r X_test')
get_ipython().magic(u'store -r y_test')
get_ipython().magic(u'store -r model')
get_ipython().magic(u'store -r scores')


# In[25]:

fig = plt.figure(figsize=(9,6))
plt.scatter(y_test, model.predict(X_test), label="predicted")
plt.plot(y_test, y_test, color='black', label="parity")
plt.title("Linear regression model prediction vs actual.\nprice = f(year, mileage), r = {0} +/- {1}".format(
    round(np.mean(scores),3), round(np.std(scores),3)), fontdict={'fontsize': 16})
plt.xlabel("Price (actual)", fontdict={'fontsize': 14})
plt.ylabel("Price (predicted)", fontdict={'fontsize': 14})
plt.legend(loc='best', fontsize='large')
plt.tight_layout()


# The model isn't great, due to anomalies for more expensive cars. This is motivation for a better model using less obvious features.

# ## Putting it together with Edmunds data
# See the Craigslist.ipynb and Edmunds.ipynb notebooks for more details. I looked at data just for Ford Focuses, due to Edmunds API limitations! Data was compared to Craigslist averages for Ford Focuses, by year.

# In[26]:

get_ipython().magic(u'store -r focus_years')
ax = focus_years.plot(y=[('price', 'mean'), 'used_private_party', 'used_tradein', 'used_tmv_retail', 'certified'], lw=3)
ax.set_ylabel("Price ($)", fontdict={'fontsize': 14})
ax.set_yticklabels(range(-2000,18000, 2000), fontdict={'fontsize': 12})
ax.set_xlabel("Year", fontdict={'fontsize': 14})
ax.set_xticklabels(range(2000,2018,2), fontdict={'fontsize': 12})
ax.set_title("Ford Focus pricing, including Craigslist averages and Edmunds data, by year", fontdict={'fontsize': 16})
ax.legend(fontsize = 'large')
fig=ax.get_figure()
fig.set_size_inches(9,6)


# In[29]:

get_ipython().magic(u'store -r focus_data')
focus_data[focus_data['year']==2016]


# In[30]:

focus_data[focus_data['year']==2014]


# In[ ]:



