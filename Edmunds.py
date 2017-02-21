
# coding: utf-8

# # Accessing data from Edmunds API

# Imports

# In[120]:

import requests
import json


# ## Get all the data on car makes

# In[19]:

f = open('api.txt', 'r')
key = f.read()
f.close()
parameters = {'api_key': key, 'fmt': 'json'}
response = requests.get("https://api.edmunds.com/api/vehicle/v2/makes", params=parameters)
data = response.json()


# In[74]:

import pandas as pd
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


# In[ ]:



