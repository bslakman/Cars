
# coding: utf-8

# # Accessing data from Edmunds API

# Imports

# In[120]:

import requests
import json


# ## Get all the data on car makes

# In[50]:

f = open('api.txt', 'r')
key = f.read()
f.close()
parameters = {'api_key': key}
response = requests.get("https://api.edmunds.com/api/vehicle/v2/makes", params=parameters)
data = response.json()


# In[51]:

#print(data)


# ### Get all the style data for a particular make/model/year

# In[113]:

make_nice_name = 'ford'
model_nice_name = 'focus'
year = '2012'


# In[114]:

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



