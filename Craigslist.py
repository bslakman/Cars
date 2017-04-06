
# coding: utf-8

# In[2]:

import requests
from bs4 import BeautifulSoup
import time
import pandas as pd


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


# In[7]:

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


# In[8]:

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


# In[16]:

def scrape_all(search_params={}):
    listings = []
    base = "http://boston.craigslist.org/search/cto"
    for i in range(0, 2000, 100):
        search_params['s'] = i
        resp = requests.get(base, params=search_params, timeout=3)
        resp.raise_for_status()
        with open('sizing.txt', 'a+') as f:
            f.write(resp.content)
        f.close()
        car_results = resp.content, resp.encoding
        doc = parse(car_results[0])
        listings.extend(extract_listings(doc))
        time.sleep(3)
    
    df = pd.DataFrame(data=listings)
    
    df['mileage'] = df.apply(lambda row: get_mileage(row['description']), axis=1)
    df['price'] = df.apply(lambda row: get_price(row['price']), axis=1)
    df['region'] = df['link'].str[1:5]
    df['year'] = df.apply(lambda row: get_year(row['description']), axis=1)
    df['std_location'] = df.apply(lambda row: re.sub('[^a-z]', '', get_standard_location(row['location'])), axis=1)
    df.set_index('link', inplace=True)
    df = df.drop_duplicates()
    
    return df


# In[65]:

all_car_info = scrape_all()
print len(all_car_info)
all_car_info = all_car_info.append(scrape_all(search_params={'searchNearby': 1}))
print len(all_car_info)
all_car_info = all_car_info.drop_duplicates()
print len(all_car_info)


# In[66]:

all_car_info = all_car_info.append(scrape_all(search_params={'sort': 'pricedsc'}))
all_car_info = all_car_info.drop_duplicates()
print len(all_car_info)


# In[67]:

all_car_info = all_car_info.append(scrape_all(search_params={'sort': 'priceasc'}))
all_car_info = all_car_info.drop_duplicates()
print len(all_car_info)


# In[68]:

all_car_info = all_car_info.append(scrape_all(search_params={'auto_transmission': 1}))
all_car_info = all_car_info.drop_duplicates()
print len(all_car_info)


# In[69]:

all_car_info.head()


# In[18]:

import pandas_profiling
pandas_profiling.ProfileReport(all_car_info)


# In[19]:

all_car_info.to_csv("all_car_info.csv", encoding='utf-8')


# Methods of getting more (older) results:
# 
# -include nearby areas (searchNearby=1)
# 
# -sort by price (sort=pricedsc or sort=priceasc)
# 
# -manual transmission (auto_transmission=1)

# In[20]:

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
sns.set_style("ticks")


# In[70]:

all_car_info.plot.scatter('year', 'mileage')
plt.ylim(0,3E5)
plt.xlim(1950,)


# In[71]:

get_ipython().magic(u'store all_car_info')


# In[72]:

all_car_info.plot.scatter('mileage', 'price')
plt.xlim(0,3E5)
plt.xlabel('Mileage', fontdict={'fontsize': 14})
plt.ylim(0,1E5)
plt.ylabel('Price', fontdict={'fontsize': 14})
plt.title('Car prices vs. Mileage', fontdict={'fontsize': 16})


# In[73]:

all_car_info.plot.scatter('year', 'price')
plt.ylim(0,2E5)
plt.ylabel('Price', fontdict={'fontsize': 14})
plt.xlim(1950,2020)
plt.xlabel('Year', fontdict={'fontsize': 14})
plt.title('Car prices vs. Year', fontdict={'fontsize': 16})


# In[81]:

regions = all_car_info[all_car_info['std_location'] != ''].groupby('std_location').agg(['mean', 'count'])


# In[82]:

regions = regions[regions['price','count'] >= 25]
regions = regions[regions['mileage','count'] >= 5]


# In[89]:

regions =regions.drop('arlin')
regions.head()


# In[90]:

#regions = regions.drop('price_mileage_ratio', axis=1)
get_ipython().magic(u'store regions')


# In[91]:

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


# In[43]:

from scipy.stats import linregress


# In[44]:

print linregress(df['mileage'][~df['price'].isnull()].dropna(), df['price'][~df['mileage'].isnull()].dropna())
print linregress(df['year'][~df['price'].isnull()].dropna(), df['price'][~df['year'].isnull()].dropna())
print linregress(df['year'][~df['mileage'].isnull()].dropna(), df['mileage'][~df['year'].isnull()].dropna())


# In[45]:

def draw_regional_fig(make, model, year):
    listings = []
    make_model = "{0} {1}".format(make,model)
    min_auto_year = int(year) - 2
    max_auto_year = int(year) + 2
    if max_auto_year > 2016:
        max_auto_year = 2016
    for i in range(0, 500, 100):
        car_results = fetch(auto_make_model=make_model, min_auto_year=min_auto_year, max_auto_year=max_auto_year, s=i)
        doc = parse(car_results[0])
        listings.extend(extract_listings(doc))
    
    df = pd.DataFrame(data=listings)
    if len(df) == 0: return "No results found, check your spelling"
    df['mileage'] = df.apply(lambda row: get_mileage(row['description']), axis=1)
    df['price'] = df.apply(lambda row: get_price(row['price']), axis=1)
    df['region'] = df['link'].str[1:5]
    df['year'] = df.apply(lambda row: get_year(row['description']), axis=1)
    
    regions = df.groupby('region').mean()
    regions = regions.append(pd.Series(data={'year': np.mean(df['year']), 'price': np.mean(df['price']), 'mileage': np.mean(df['mileage'])}, name='AVERAGE'))
    
    my_title = 'Average Price and Mileage of Used {0} {1}, {2}-{3}, by region, n={4}'.format(make, model, min_auto_year, max_auto_year, len(df))
    ax = regions['price'].plot.bar(position=0, width=0.3, alpha=0.5, legend=True, title=my_title)
    ax.set_ylabel('Price($)')
    ax = regions['mileage'].plot.bar(secondary_y=True, color='green', position=1, width=0.3, alpha=0.5, legend=True)
    ax.set_ylabel('Mileage')
    sns.despine(top=True, right=False)
    fig=ax.get_figure()
    
    return fig


# In[46]:

focus_data = scrape_all(search_params={'auto_make_model': 'ford focus'})
print len(focus_data)
focus_data = focus_data.append(scrape_all(search_params={'auto_make_model': 'ford focus', 'searchNearby': 1}))
print len(focus_data)
focus_data = focus_data.drop_duplicates()
print len(focus_data)


# In[47]:

focus_data = focus_data.append(scrape_all(search_params={'auto_make_model': 'ford focus', 'sort': 'priceasc'}))
print len(focus_data)
focus_data = focus_data.drop_duplicates()
print len(focus_data)


# In[48]:

focus_data = focus_data.append(scrape_all(search_params={'auto_make_model': 'ford focus', 'sort': 'pricedsc'}))
print len(focus_data)
focus_data = focus_data.drop_duplicates()
print len(focus_data)


# In[49]:

focus_data = focus_data.append(scrape_all(search_params={'auto_make_model': 'ford focus', 'auto_transmission': 1}))
print len(focus_data)
focus_data = focus_data.drop_duplicates()
print len(focus_data)


# In[50]:



# In[51]:

focus_years = focus_data.groupby('year').agg(['mean', 'count'])
focus_years


# In[52]:



# In[53]:

get_ipython().magic(u'store -r ford_focus_years')


# In[54]:

focus_years = focus_years.join(ford_focus_years)
focus_years


# In[55]:

get_ipython().magic(u'store focus_years')


# In[56]:

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


# In[ ]:



# In[ ]:



# In[57]:

get_ipython().magic(u'store focus_data')


# In[58]:

from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import ShuffleSplit, train_test_split


# In[100]:

data = all_car_info[['year', 'mileage', 'price']].dropna()
data = data[data['price'] < 100000]
data = data[data['price'] > 99]
data = data[data['mileage'] < 500000]
data = data[data['year'] > 1986]
X = data[['year', 'mileage']]
y = data['price']


# In[182]:

coeff = []
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    coeff.append(model.coef_)
    scores.append(model.score(X_test,y_test))
print "Average score = {0} +/- {1}".format(round(np.mean(scores),3), round(np.std(scores),3))


# In[183]:

get_ipython().magic(u'store X_test')
get_ipython().magic(u'store y_test')
get_ipython().magic(u'store model')
get_ipython().magic(u'store scores')


# In[184]:

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


# In[185]:

print model.coef_


# Try and turn the plot into a Bokeh plot...

# In[170]:

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import Axis, HoverTool


# In[186]:

radii = (X_test['year']-1986)/20 * 1500 # size of points is scaled to year


# In[172]:

output_notebook()


# In[194]:

hover = HoverTool(tooltips=[('Predicted, Actual', '$x, $y')])
ax_limit = max(y_test + model.predict(X_test)) + 1000
p = figure(x_range=(0,ax_limit), y_range=(0,ax_limit), plot_width=500, plot_height=400, tools=[hover])
for axis in p.select(dict(type=Axis)):
    axis.formatter.use_scientific = False
p.circle(y_test, model.predict(X_test), radius = radii, line_color='black', fill_alpha=0.5)
p.line(y_test, y_test, color = 'gray')
show(p)


# In[ ]:
all_car_info = pd.read_csv('all_car_info.csv')
all_car_info.head()

from ediblepickle import checkpoint
import os.path
import time
import sys
import re

sys.setrecursionlimit(5000)

cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

@checkpoint(key=lambda args, kwargs: "-".join(args[0].split('.')[0].split('/')[-3:]) + '.p', work_dir=cache_dir)
def get_page_text(path):
    response = requests.get('https://boston.craigslist.org' + path)
    time.sleep(3)
    return response.text

attr_list= []
for link in all_car_info['link']:
    attributes = {'link' : link}
    soup = BeautifulSoup(get_page_text(link), "lxml")
    map_and_attr = soup.select("div.mapAndAttrs > p.attrgroup")
    try:
        attributes['year_make_model'] = map_and_attr[0].select("span > b")[0].get_text()
    except:
        pass
    try:
        listing_attrs = map_and_attr[1]
        for item in listing_attrs(text=re.compile('.+:')):
            attributes[item.split(':')[0]] = item.parent.select('b')[0].get_text()
    except:
        pass
    attr_list.append(attributes)


len(attr_list)
get_ipython().magic(u'store attr_list')

attr_df = pd.DataFrame(data=attr_list)
attr_df.set_index('link', inplace=True)
get_ipython().magic(u'store attr_df')
attr_df[3:8]

all_car_info = pd.merge(all_car_info, attr_df, on='link')
