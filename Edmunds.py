
# coding: utf-8

# In[38]:

import requests
import json


# In[45]:

f = open('api.txt', 'r')
key = f.read()
f.close()
parameters = {'api_key': key}
response = requests.get("https://api.edmunds.com/api/vehicle/v2/:makes?", params=parameters)
data = response.json()


# In[46]:

nice_name = data['status']
parameters = {'makeNiceName': nice_name, 'year': 2012, 'fmt': json}
civic_data = requests.get("https://api.edmunds.com/api/vehicle/v2/:makeNiceName?api_key=2e7hjvffdg4x4mvrw2vtgvad", params=parameters).json()


# In[47]:

print(civic_data)


# craigslist price - kelly blue book value = dealer price - what they'll give you for it (percentage). give a range of uncertainties...

# In[ ]:



