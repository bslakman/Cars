
# coding: utf-8

# In[36]:

import requests
from bs4 import BeautifulSoup
import time


# In[2]:

def fetch(query = None, auto_make_model = None, min_auto_year = None, max_auto_year = None, s=0):
    search_params = {key: val for key, val in locals().items() if val is not None}
    if not search_params: 
        raise ValueError("No valid keywords")
        
    base = "http://boston.craigslist.org/search/cto"
    resp = requests.get(base, params=search_params, timeout=3)
    resp.raise_for_status()
    return resp.content, resp.encoding


# In[3]:

def parse(html, encoding='utf-8'):
    parsed = BeautifulSoup(html, 'lxml', from_encoding=encoding)
    return parsed


# In[4]:

def extract_listings(parsed):
    listings = parsed.find_all('p', class_='result-info')
    extracted = []
    for listing in listings:
        title = listing.find('a', class_='result-title hdrlnk')
        price = listing.find('span', class_='result-price')
        try:
            price_string = price.string.strip()
        except AttributeError:
            price_string = ''
        location = listing.find('span', class_='result-hood')
        try:
            loc_string = location.string.strip()[1:-1].split()[0]
        except AttributeError:
            loc_string = ''
        this_listing = {
            'link': title.attrs['href'],
            'description': title.string.strip(),
            'price': price_string,
            'location': loc_string
        }
        extracted.append(this_listing)
    return extracted


# In[5]:

import pandas as pd
import numpy as np


# In[6]:

import re

def get_mileage(description):
    description = description.lower().split('k miles')
    if len(description) == 1:
        description = description[0].split('000 miles')
        if len(description) == 1:
            try:
                description = re.search('(\d{1,3})k', description[0]).groups()
            except:
                return np.nan
    mileage = re.sub('[^0-9]', '', description[0].split()[-1])
    try:
        mileage = int(mileage) * 1000
        return mileage
    except:
        return np.nan


# In[16]:

def get_year(description):
    description = re.split('(20[0-9][0-9])', description)
    if len(description) == 1:
        description = re.split('(19[0-9][0-9])', description[0])
    if len(description) == 1:
        description = re.split('([0-1][0-9])', description[0])
    if len(description) == 1:
        return np.nan
    if len(description[1]) == 4: 
        year = description[1]
    elif int(description[1]) > 17: 
        year = '19' + description[1]
    else: 
        year = '20' + description[1]
    try:
        return int(year) if int(year) <= 2017 else np.nan
    except:
        return np.nan


# In[24]:

def get_standard_location(location):
    """
    Use first 5 characters of location in order to group. Gets rid of much of the weird stuff
    """
    if len(location) < 5:
        return re.sub('[^a-z]', '', location.lower())
    else:
        return re.sub('[^a-z]', '', location[:5].lower())


# In[9]:

def get_price(price):
    try:
        return int(price[1:]) if int(price[1:]) > 100 else np.nan
    except:
        return np.nan


# In[111]:

def scrape_all(search_params={}):
    listings = []
    base = "http://boston.craigslist.org/search/cto"
    for i in range(0, 1000, 100):
        search_params['s'] = i
        resp = requests.get(base, params=search_params, timeout=3)
        resp.raise_for_status()
        with open('sizing.txt', 'a+') as f:
            f.write(resp.content)
        f.close()
        car_results = resp.content, resp.encoding
        doc = parse(car_results[0])
        listings.extend(extract_listings(doc))
        time.sleep(2)
    
    df = pd.DataFrame(data=listings)
    
    df['mileage'] = df.apply(lambda row: get_mileage(row['description']), axis=1)
    df['price'] = df.apply(lambda row: get_price(row['price']), axis=1)
    df['region'] = df['link'].str[1:5]
    df['year'] = df.apply(lambda row: get_year(row['description']), axis=1)
    df['std_location'] = df.apply(lambda row: re.sub('[^a-z]', '', get_standard_location(row['location'])), axis=1)
    df.set_index('link', inplace=True)
    df = df.drop_duplicates()
    
    return df


# In[108]:

all_car_info = scrape_all()
print len(all_car_info)
all_car_info = all_car_info.append(scrape_all(search_params={'searchNearby': 1}))
print len(all_car_info)
all_car_info = all_car_info.drop_duplicates()
print len(all_car_info)


# In[109]:

all_car_info = all_car_info.append(scrape_all(search_params={'sort': 'pricedsc'}))
all_car_info = all_car_info.drop_duplicates()
print len(all_car_info)


# In[110]:

all_car_info = all_car_info.append(scrape_all(search_params={'sort': 'priceasc'}))
all_car_info = all_car_info.drop_duplicates()
print len(all_car_info)


# In[112]:

all_car_info = all_car_info.append(scrape_all(search_params={'auto_transmission': 1}))
all_car_info = all_car_info.drop_duplicates()
print len(all_car_info)


# In[113]:

all_car_info['std_location'] = all_car_info.apply(lambda row: re.sub('[^a-z]', '', row['std_location']), axis=1)
all_car_info.head()


# In[245]:

all_car_info.set_index('link', inplace=True)
all_car_info.head()


# In[246]:

import pandas_profiling
pandas_profiling.ProfileReport(all_car_info)


# In[248]:

all_car_info.to_csv("all_car_info.csv", encoding='utf-8')


# Methods of getting more (older) results:
# 
# -include nearby areas (searchNearby=1)
# 
# -sort by price (sort=pricedsc or sort=priceasc)
# 
# -manual transmission (auto_transmission=1)

# In[249]:

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
sns.set_style("ticks")


# In[250]:

all_car_info.plot.scatter('year', 'mileage')
plt.ylim(0,3E5)
plt.xlim(1950,)


# In[569]:

get_ipython().magic(u'store all_car_info')


# In[568]:

all_car_info.plot.scatter('mileage', 'price')
plt.xlim(0,3E5)
plt.xlabel('Mileage', fontdict={'fontsize': 14})
plt.ylim(0,1E5)
plt.ylabel('Price', fontdict={'fontsize': 14})
plt.title('Car prices vs. Mileage', fontdict={'fontsize': 16})


# In[565]:

all_car_info.plot.scatter('year', 'price')
plt.ylim(0,2E5)
plt.ylabel('Price', fontdict={'fontsize': 14})
plt.xlim(1950,2020)
plt.xlabel('Year', fontdict={'fontsize': 14})
plt.title('Car prices vs. Year', fontdict={'fontsize': 16})


# In[147]:

print all_car_info[all_car_info['price'] >= 150000]


# In[ ]:

fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Year')
ax1.set_ylabel('Price($)')
ax1.set_title('Price vs Mileage and Year for Used Honda Civics, 2000-2016', y= 1.2)
plt.plot(df['year'], df['price'], '.', ms=10, label='year')
ax1.set_xbound(lower=1999, upper=2017)
ax1.legend(loc='best')
ax2 = ax1.twiny()
ax2.set_xlabel('Mileage')
plt.plot(df['mileage'], df['price'], 'g*', ms=10, label='mileage')
ax2.legend(loc=2)
plt.tight_layout()
plt.savefig('price_year_mileage.pdf')


# In[ ]:

plt.plot(df['price'], df['mileage'], linestyle='', marker='.')


# In[263]:

regions = all_car_info[all_car_info['std_location'] != ''].groupby('std_location').agg(['mean', 'count'])


# In[264]:

regions = regions[regions['price','count'] >= 50]
regions = regions[regions['mileage','count'] >= 5]


# In[265]:

regions.head()


# In[570]:

get_ipython().magic(u'store regions')


# In[549]:

regions.sort_values(by=[('price', 'mean')], inplace=True)
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
fig.savefig('price_mileage_region.pdf', bbox_inches='tight')


# In[ ]:

from scipy.stats import linregress


# In[ ]:

print linregress(df['mileage'][~df['price'].isnull()].dropna(), df['price'][~df['mileage'].isnull()].dropna())
print linregress(df['year'][~df['price'].isnull()].dropna(), df['price'][~df['year'].isnull()].dropna())
print linregress(df['year'][~df['mileage'].isnull()].dropna(), df['mileage'][~df['year'].isnull()].dropna())


# In[ ]:

focus_data = scrape_all(search_params={'auto_make_model': 'ford focus'})
print len(focus_data)
focus_data = focus_data.append(scrape_all(search_params={'auto_make_model': 'ford focus', 'searchNearby': 1}))
print len(focus_data)
focus_data = focus_data.drop_duplicates()
print len(focus_data)


# In[287]:

focus_data = focus_data.append(scrape_all(search_params={'auto_make_model': 'ford focus', 'sort': 'priceasc'}))
print len(focus_data)
focus_data = focus_data.drop_duplicates()
print len(focus_data)


# In[288]:

focus_data = focus_data.append(scrape_all(search_params={'auto_make_model': 'ford focus', 'sort': 'pricedsc'}))
print len(focus_data)
focus_data = focus_data.drop_duplicates()
print len(focus_data)


# In[289]:

focus_data = focus_data.append(scrape_all(search_params={'auto_make_model': 'ford focus', 'auto_transmission': 1}))
print len(focus_data)
focus_data = focus_data.drop_duplicates()
print len(focus_data)


# In[290]:

focus_data


# In[291]:

focus_years = focus_data.groupby('year').agg(['mean', 'count'])
focus_years


# In[293]:

focus_data[focus_data['year']==2016]


# In[432]:

get_ipython().magic(u'store -r ford_focus_years')


# In[433]:

focus_years = focus_years.join(ford_focus_years)
focus_years


# In[574]:

get_ipython().magic(u'store focus_years')


# In[522]:

get_ipython().magic(u'matplotlib inline')
focus_years = focus_years.replace(0, np.nan)
ax = focus_years.plot(y=[('price', 'mean'), 'used_private_party', 'used_tradein', 'used_tmv_retail', 'certified'], lw=3)
ax.set_ylabel("Price ($)", fontdict={'fontsize': 14})
ax.set_yticklabels(range(-2000,18000, 2000), fontdict={'fontsize': 12})
ax.set_xlabel("Year", fontdict={'fontsize': 14})
ax.set_xticklabels(range(2000,2018,2), fontdict={'fontsize': 12})
ax.set_title("Ford Focus pricing, including Craigslist averages and Edmunds data, by year", fontdict={'fontsize': 16})
ax.legend(fontsize = 'large')
fig=ax.get_figure()
fig.set_size_inches(9,6)
fig.savefig('compare_prices.pdf', bbox_inches='tight')


# In[457]:

focus_data[focus_data['year']==2016]


# In[459]:

focus_data[focus_data['year']==2014]


# In[575]:

get_ipython().magic(u'store focus_data')


# In[356]:

from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import ShuffleSplit, train_test_split


# In[383]:

data = all_car_info[['year', 'mileage', 'price']].dropna()
data = data[data['price'] < 100000]
data = data[data['mileage'] < 500000]
data = data[data['year'] > 1986]
X = data[['year', 'mileage']]
y = data['price']


# In[398]:

coeff = []
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    coeff.append(model.coef_)
    scores.append(model.score(X_test,y_test))
print "Average score = {0} +/- {1}".format(round(np.mean(scores),3), round(np.std(scores),3))


# In[410]:
# In[573]:

get_ipython().magic(u'store X_test')
get_ipython().magic(u'store y_test')
get_ipython().magic(u'store model')
get_ipython().magic(u'store scores')

fig = plt.figure(figsize=(9,6))
plt.scatter(y_test, model.predict(X_test), label="predicted")
plt.plot(y_test, y_test, color='black', label="parity")
plt.title("Linear regression model prediction vs actual.\nprice = f(year, mileage), r = {0} +/- {1}".format(
    round(np.mean(scores),3), round(np.std(scores),3)), fontdict={'fontsize': 16})
plt.xlabel("Price (actual)", fontdict={'fontsize': 14})
plt.ylabel("Price (predicted)", fontdict={'fontsize': 14})
plt.legend(loc='best', fontsize='large')
plt.tight_layout()
fig.savefig('regression.pdf')


# In[402]:

print model.coef_


# In[ ]:



