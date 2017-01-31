
# coding: utf-8

# In[32]:

import requests
from bs4 import BeautifulSoup


# In[24]:

def fetch(query = None, auto_make_model = None, min_auto_year = None, max_auto_year = None):
    search_params = {key: val for key, val in locals().items() if val is not None}
    if not search_params: 
        raise ValueError("No valid keywords")
        
    base = "http://boston.craigslist.org/search/cto"
    resp = requests.get(base, params=search_params, timeout=3)
    resp.raise_for_status()
    return resp.content, resp.encoding


# In[25]:

car_results = fetch(auto_make_model="honda civic", min_auto_year=2010, max_auto_year=2014)


# In[33]:

def parse(html, encoding='utf-8'):
    parsed = BeautifulSoup(html, from_encoding=encoding)
    return parsed


# In[90]:

def extract_listings(parsed):
    #title_attr = {'data-id'}
    listings = parsed.find_all('p', class_='result-info')
    extracted = []
    for listing in listings:
        title = listing.find('a')
        title = {key: listing.attrs.get(key, '') for key in title_attr}
    #    this_listing = {
    #        'ID': title,
    #    }
    #    extracted.append(this_listing)
    return listings #extracted


# In[91]:

doc = parse(car_results[0])
#print(doc.prettify())


# In[92]:

listings = extract_listings(doc)
print len(listings)
print listings[5]


# In[ ]:



