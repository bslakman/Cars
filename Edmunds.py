
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


# In[78]:

ford_fiesta_years = pd.DataFrame(data=ford_models.loc['Ford_Fiesta']['years'])
ford_fiesta_years.set_index('year', inplace=True)
print len(ford_fiesta_years)
ford_fiesta_years


# In[79]:

car_makes.to_csv('car_makes.csv')


# ## Get all the styles for all the years of a particular make/model combination

# In[57]:

import time
makeNiceName = 'ford'
modelNiceName = 'fiesta'
fiesta_style_details = []
parameters = {}
parameters = {'api_key': key, 'fmt': 'json'}
for year in ford_fiesta_years.index:
    fiesta_style_details.append(requests.get("https://api.edmunds.com/api/vehicle/v2/{0}/{1}/{2}/styles".format(makeNiceName,modelNiceName,year), params=parameters))
    time.sleep(3)


# In[85]:

ford_fiesta_years = ford_fiesta_years.loc[2011:2016]
ford_fiesta_years['styles'] = fiesta_style_details


# In[86]:

ford_fiesta_years


# ### Get all the style data for a particular make/model/year

# In[89]:

ford_fiesta_styles_2011 = pd.DataFrame(data=ford_fiesta_years.loc[2011]['styles']['styles'])
ford_fiesta_styles_2011.set_index('id', inplace=True)
ford_fiesta_styles_2011


# In[3]:

make_nice_name = 'kia'
model_nice_name = 'sportage'
year = '2012'


# In[16]:

parameters = {'fmt': 'json', 'api_key': key}
model_data = requests.get("https://api.edmunds.com/api/vehicle/v2/{0}/{1}/{2}/styles".format(make_nice_name,model_nice_name,year), params=parameters).json()
#print(civic_data)


# craigslist price - kelly blue book value = dealer price - what they'll give you for it (percentage). give a range of uncertainties...

# In[116]:

for x in model_data['styles']:
    if x['submodel']['body']=='Sedan': print(x['name'].encode('utf-8'))


# ## Get price data for a particular style of that vehicle in a given zip code

# In[117]:

style = model_data['styles'][0] # chose one randomly
print(style)
for k,v in style.items():
    if isinstance(v, dict):
        v = {str(x).encode('utf-8'): str(y).encode('utf-8') for x,y in v.items()}
    print "{0}: {1}".format(k,v)
styleid = style['id']
zip_code = '02143'


# In[118]:

parameters = {'styleid': styleid, 'zip': zip_code, 'fmt' : 'json', 'api_key': key}
typical_data = requests.get("https://api.edmunds.com/v1/api/tmv/tmvservice/calculatetypicallyequippedusedtmv", params=parameters).json()
certified_data = requests.get("https://api.edmunds.com/v1/api/tmv/tmvservice/findcertifiedpriceforstyle", params=parameters)


# In[119]:

print(typical_data['tmv']['totalWithOptions']['usedPrivateParty'])
print(typical_data['tmv']['totalWithOptions']['usedTradeIn'])
print(certified_data.content)


# In[ ]:



