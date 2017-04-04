
# coding: utf-8

# # Accessing data from Edmunds API

# Imports

# In[120]:

import requests
import json
import pandas as pd
import numpy as np
import re
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns


# ## Get all the data on car makes

# In[19]:

f = open('api.txt', 'r')
key = f.read()
f.close()
parameters = {'api_key': key, 'fmt': 'json'}
response = requests.get("https://api.edmunds.com/api/vehicle/v2/makes", params=parameters)
data = response.json()


# In[74]:

car_makes = pd.DataFrame(data=data['makes'])
car_makes.set_index('id', inplace=True)
print len(car_makes)
car_makes.head(20)


# In[77]:

ford_models = pd.DataFrame(data=car_makes.loc[200005143]['models'])
ford_models.set_index('id', inplace=True)
print len(ford_models)
ford_models


# In[102]:

ford_focus_years = pd.DataFrame(data=ford_models.loc['Ford_Focus']['years'])
ford_focus_years.set_index('year', inplace=True)
print len(ford_focus_years)
ford_focus_years


# In[96]:

ford_focus_years.index[:-1]


# In[79]:

car_makes.to_csv('car_makes.csv')


# ## Get all the styles for all the years of a particular make/model combination

# In[98]:

import time
makeNiceName = 'ford'
modelNiceName = 'focus'
focus_style_details = []
parameters = {}
parameters = {'api_key': key, 'fmt': 'json'}
for year in ford_focus_years.index[:-1]: # don't include 2017
    focus_style_details.append(requests.get("https://api.edmunds.com/api/vehicle/v2/{0}/{1}/{2}/styles".format(makeNiceName,modelNiceName,year), params=parameters))
    time.sleep(3)


# In[124]:

ford_focus_years = ford_focus_years.loc[2000:2016]
ford_focus_years['styles'] = focus_style_details


# In[125]:

ford_focus_years['styles'] = ford_focus_years.apply(lambda row: row['styles'].json(), axis=1)


# In[126]:

ford_focus_years


# ### Get all the style data for a particular make/model/year

# In[136]:

ford_focus_styles_2016 = pd.DataFrame(data=ford_focus_years.loc[2016]['styles']['styles'])
ford_focus_styles_2016.set_index('id', inplace=True)
ford_focus_styles_2016


# craigslist price - kelly blue book value = dealer price - what they'll give you for it (percentage). give a range of uncertainties...

# ## Get price data for a particular style of that vehicle in a given zip code

# In[150]:

zip_code = '02115'
data = []
for year in ford_focus_years.index:
    style_list = ford_focus_years.loc[year]['styles']['styles']
    possible_styles = [style for style in style_list if 'se 4dr sedan' in style['name'].lower()]
    try:
        styleid = possible_styles[0]['id']
    except:
        break
    parameters = {'styleid': styleid, 'zip': zip_code, 'fmt' : 'json', 'api_key': key}
    typical_data = requests.get("https://api.edmunds.com/v1/api/tmv/tmvservice/calculatetypicallyequippedusedtmv", params=parameters).json()
    time.sleep(2)
    data.append(typical_data)


# In[140]:

print(typical_data['tmv']['totalWithOptions']['usedPrivateParty'])
print(typical_data['tmv']['totalWithOptions']['usedTradeIn'])
#print(certified_data.content)


# In[152]:

ford_focus_years['typical_data'] = data


# In[170]:

ford_focus_years['used_private_party'] = ford_focus_years.apply(lambda row: row['typical_data']['tmv']['totalWithOptions']['usedPrivateParty'], axis=1)
ford_focus_years['used_tradein'] = ford_focus_years.apply(lambda row: row['typical_data']['tmv']['totalWithOptions']['usedTradeIn'], axis=1)
ford_focus_years['used_tmv_retail'] = ford_focus_years.apply(lambda row: row['typical_data']['tmv']['totalWithOptions']['usedTmvRetail'], axis=1)
ford_focus_years['certified'] = ford_focus_years.apply(lambda row: row['typical_data']['tmv']['certifiedUsedPrice'], axis=1)


# In[171]:

ford_focus_years


# In[172]:

get_ipython().magic(u'store ford_focus_years')


from ediblepickle import checkpoint
import os.path
import time

cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
    
@checkpoint(key=lambda args, kwargs: "-".join(args) + '.p', work_dir=cache_dir)
def get_styles(make, model, year):
    parameters = {'api_key': key, 'fmt': 'json'}
    styles = requests.get("https://api.edmunds.com/api/vehicle/v2/{0}/{1}/{2}/styles".format(make,model,year), params=parameters)
    time.sleep(2)
    return styles.json()

@checkpoint(key=lambda args, kwargs: "-".join(args[:4]) + '.p', work_dir=cache_dir)
def get_typical_data(make, model, year, zip_code, styles):
    try:
        possible_styles = [style for style in styles['styles'] if '4dr sedan' in style['name'].lower()]
    except:
        print styles
        return None
    try:
        styleid = possible_styles[0]['id']
    except: 
        print make, model, year
        return None
    parameters = {'styleid': styleid, 'zip': zip_code, 'fmt' : 'json', 'api_key': key}
    typical_data = requests.get("https://api.edmunds.com/v1/api/tmv/tmvservice/calculatetypicallyequippedusedtmv", params=parameters).json()
    time.sleep(2)
    return typical_data

def extract_typical_data(make, model, year, zip_code):
    typical_data = get_typical_data(make, model, year, zip_code, get_styles(make, model, year))
    typical_data_tmv = typical_data['tmv']
    try:
        typical_data_tmv = typical_data['tmv']
    except:
        print make, model, year
    data_dict = {'used_private_party': typical_data_tmv['totalWithOptions']['usedPrivateParty'],
                'used_tradein': typical_data_tmv['totalWithOptions']['usedTradeIn'],
                'used_tmv_retail': typical_data_tmv['totalWithOptions']['usedTmvRetail'],
                'certified': typical_data_tmv['certifiedUsedPrice']}
    return data_dict
car_makes = pd.read_csv('car_makes.csv').set_index('niceName')
extract_typical_data('hyundai', 'accent', '1997', '02143')
years = ['1997', '2006', '2015']
make_model = [('hyundai', 'accent'),('ford', 'taurus'),('mercedes-benz', 'e-class')]
pricing_data = {}

for make, model in make_model:
    for year in years:
        pricing_data[(make, model, year)] = extract_typical_data(make, model, year, '02143')

pricing_df = pd.DataFrame(data=pricing_data)
pricing_df = pricing_df.transpose()
pricing_df['retail_markup'] = pricing_df.apply(
    lambda row: (row['used_tmv_retail']-row['used_tradein']), axis=1)

pricing_df['retail_markup_pct'] = pricing_df.apply(
    lambda row: 100.0 * (row['used_tmv_retail']-row['used_tradein']) / row['used_tradein'], axis=1)

pricing_df['certified_markup_pct'] = pricing_df.apply(
    lambda row: 100.0 * (row['certified']-row['used_tradein']) / row['used_tradein'], axis=1)


# In[16]:

pricing_df


# In[102]:

get_ipython().magic(u'store pricing_df')



get_ipython().magic(u'matplotlib inline')
import seaborn as sns
get_ipython().magic(u'store -r pricing_df')

color_dict = {'hyundai': sns.color_palette()[0], 'ford': sns.color_palette()[1], 'mercedes-benz': sns.color_palette()[2]}
fill_dict = {'1997': 'x', '2006': '+', '2015': '.'}
pricing_df['certified_markup_pct'] = pricing_df.apply(lambda row: row['certified_markup_pct'] if row['certified_markup_pct'] > 0 else np.nan, axis=1)
fig.add_subplot(1,2,1)
ax1 = pricing_df.set_index('used_private_party', append=True).sort_index(level=(2,3,1)).reset_index(
    level=3, drop=True).plot.bar(y=['retail_markup_pct', 'certified_markup_pct'], width=0.75)
plt.ylabel('Markup (% of TMV)')
L=plt.legend()
L.get_texts()[0].set_text('Retail')
L.get_texts()[1].set_text('Certified Pre-Owned')
fig.add_subplot(1,2,2)
ax2 = pricing_df.set_index('used_private_party', append=True).sort_index(
    level=(2,3,1)).reset_index(level=3, drop=True).plot.bar(y=['retail_markup'], width=0.75, legend=None)
plt.ylabel('Markup ($)')
#L2=plt.legend()
#L2.get_texts()[0].set_text('Retail')
for patch, hatch, color in zip(ax2.containers[0], [fill_dict[i[2]] for i in pricing_df.index], [color_dict[j[0]] for j in pricing_df.index]):
    patch.set_hatch(hatch)
    patch.set_color(color)


# In[ ]:



