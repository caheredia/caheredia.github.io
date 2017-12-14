
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc" style="margin-top: 1em;"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Data-Exploration" data-toc-modified-id="Data-Exploration-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Exploration</a></span><ul class="toc-item"><li><span><a href="#Join-tables-in-a-meaningful-way" data-toc-modified-id="Join-tables-in-a-meaningful-way-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Join tables in a meaningful way</a></span></li><li><span><a href="#Adding-external-information-to-augment-table" data-toc-modified-id="Adding-external-information-to-augment-table-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Adding external information to augment table</a></span></li><li><span><a href="#Cluster-data-into-at-least-3-and-at-most-20-groups" data-toc-modified-id="Cluster-data-into-at-least-3-and-at-most-20-groups-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Cluster data into at least 3 and at most 20 groups</a></span></li><li><span><a href="#Data-Visualization" data-toc-modified-id="Data-Visualization-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Data Visualization</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Conclusion</a></span><ul class="toc-item"><li><span><a href="#Insights-to-advance-a-company-financially" data-toc-modified-id="Insights-to-advance-a-company-financially-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Insights to advance a company financially</a></span></li></ul></li></ul></div>

# Introduction 
Depending on the business, often sales data is recorded in a nonstandard form. But in order to run analytics on sales data, raw data files need to be cleaned up and merged together in a meaningful way. Therefore, this Jupyter notebook merges two CSV files (one addressing delivery times associated with geoinformation, the other individual sales). The notebook unpacked many fields from geoinformation such as zip code and city and average income. It also assessed metrics by graphing extrapolated data. The data revealed that most of the sales are concentrated in large cities: San Francisco, Oakland, San Diego. The findings reveal that bay area business strategies are effective relative to the rest of the state. The heatmap graph of sales reveals many potential markets for growth. 


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from uszipcode import ZipcodeSearchEngine
search = ZipcodeSearchEngine()
%matplotlib inline
import os
import gmaps
from ipywidgets.embed import embed_minimal_html

gmaps.configure(api_key=os.environ["GOOGLE_API_KEY"])
```

# Data Exploration

## Join tables in a meaningful way 
Below I'll join the tables with merge on the column 'order id'. It should be noted that the tables are not of equal size, thus some of the features (columns) will have missing values. 


```python
df_delivery = pd.read_csv('../data/delivery_geography.csv')
df_delivery.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>orderid</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>estimateddeliverytime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1552679</td>
      <td>37.697073</td>
      <td>-122.485903</td>
      <td>259.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1552697</td>
      <td>37.782185</td>
      <td>-122.454544</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1552675</td>
      <td>37.780840</td>
      <td>-122.395820</td>
      <td>228.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1552934</td>
      <td>37.783131</td>
      <td>-122.388962</td>
      <td>209.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1552983</td>
      <td>37.323786</td>
      <td>-121.878904</td>
      <td>345.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_prices = pd.read_csv('../data/delivery_prices.csv')
df_prices.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>orderid</th>
      <th>deliveredat</th>
      <th>totalprice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1552553</td>
      <td>2017-08-08T00:02:29.980+00:00</td>
      <td>176.11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1552559</td>
      <td>2017-08-08T00:09:06.077+00:00</td>
      <td>71.03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1552499</td>
      <td>2017-08-08T00:15:03.847+00:00</td>
      <td>61.95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1552617</td>
      <td>2017-08-08T00:16:40.820+00:00</td>
      <td>62.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1552632</td>
      <td>2017-08-08T00:31:27.957+00:00</td>
      <td>64.16</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change the index to 'orderid' column
df_prices = df_prices.set_index('orderid')
df_delivery = df_delivery.set_index('orderid')
# Merge on 'oderid' column
df_join = pd.merge(df_prices, df_delivery, how='outer',
                   left_index=True, right_index=True)
# rename columns
df_join.columns = ['date', 'sales', 'lat', 'lon', 'delivery time']
# sample of table values
df_join[1020:1025]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>sales</th>
      <th>lat</th>
      <th>lon</th>
      <th>delivery time</th>
    </tr>
    <tr>
      <th>orderid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1553535</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.012453</td>
      <td>-118.457838</td>
      <td>1077.0</td>
    </tr>
    <tr>
      <th>1553536</th>
      <td>2017-08-08T02:26:01.213+00:00</td>
      <td>43.73</td>
      <td>37.779351</td>
      <td>-122.497121</td>
      <td>235.0</td>
    </tr>
    <tr>
      <th>1553537</th>
      <td>2017-08-08T02:30:57.383+00:00</td>
      <td>137.47</td>
      <td>37.807403</td>
      <td>-122.301621</td>
      <td>172.0</td>
    </tr>
    <tr>
      <th>1553538</th>
      <td>2017-08-08T02:38:09.680+00:00</td>
      <td>56.94</td>
      <td>37.748804</td>
      <td>-122.423628</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>1553539</th>
      <td>2017-08-08T03:30:50.513+00:00</td>
      <td>65.22</td>
      <td>37.485039</td>
      <td>-122.192457</td>
      <td>119.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
'Number of rows for df_join dataframe:  {}'.format(len(df_join))
```




    'Number of rows for df_join dataframe:  3849'



## Adding external information to augment table

- Using the uszipcode 0.1.3 python package I can look up the zipcode, city, and average income for each latitude, longitude entry
- Using the Date Functionality in Pandas will allow the unpacking of the time stamp into days and time, which will give a more granular view of when purchases occur. 

Even though the sample data did not vary in weekdays this information would be a good metric to know for market exploration purposes.



```python
def geo_data(df, lat, lon):
    '''This function returns the dataframe with zipcode, average income, and city as columns. 
    The function takes a dataframe and column names as strings for latitude and longitude data.
    It uses the uszipcode library to convert latitude, longitude into zipcode, city, and avg income
    for geo data.'''
    # average income
    df['avg income'] = df.apply(lambda x: search.by_coordinate(x[lat], x[lon], radius=4, returns=1)[
        0]['Wealthy'] if (x[lat] > 0) else None, axis=1).round(0)
    # zipcode
    df['zipcode'] = df.apply(lambda x:  search.by_coordinate(
        x[lat], x[lon], radius=4, returns=1)[0]['Zipcode'] if (x[lat] > 0) else None, axis=1)
    # city
    df['city'] = df.apply(lambda x:  search.by_coordinate(
        x[lat], x[lon], radius=4, returns=1)[0]['City'] if (x[lat] > 0) else None, axis=1)
```


```python
geo_data(df_join, 'lat', 'lon')
df_join[1020:1025]  #sample of table values 
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>sales</th>
      <th>lat</th>
      <th>lon</th>
      <th>delivery time</th>
      <th>avg income</th>
      <th>zipcode</th>
      <th>city</th>
    </tr>
    <tr>
      <th>orderid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1553535</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.012453</td>
      <td>-118.457838</td>
      <td>1077.0</td>
      <td>39765.0</td>
      <td>90405</td>
      <td>Santa Monica</td>
    </tr>
    <tr>
      <th>1553536</th>
      <td>2017-08-08T02:26:01.213+00:00</td>
      <td>43.73</td>
      <td>37.779351</td>
      <td>-122.497121</td>
      <td>235.0</td>
      <td>29058.0</td>
      <td>94121</td>
      <td>San Francisco</td>
    </tr>
    <tr>
      <th>1553537</th>
      <td>2017-08-08T02:30:57.383+00:00</td>
      <td>137.47</td>
      <td>37.807403</td>
      <td>-122.301621</td>
      <td>172.0</td>
      <td>14127.0</td>
      <td>94607</td>
      <td>Oakland</td>
    </tr>
    <tr>
      <th>1553538</th>
      <td>2017-08-08T02:38:09.680+00:00</td>
      <td>56.94</td>
      <td>37.748804</td>
      <td>-122.423628</td>
      <td>20.0</td>
      <td>27502.0</td>
      <td>94110</td>
      <td>San Francisco</td>
    </tr>
    <tr>
      <th>1553539</th>
      <td>2017-08-08T03:30:50.513+00:00</td>
      <td>65.22</td>
      <td>37.485039</td>
      <td>-122.192457</td>
      <td>119.0</td>
      <td>14781.0</td>
      <td>94063</td>
      <td>Redwood City</td>
    </tr>
  </tbody>
</table>
</div>




```python
def df_day(df, timestamp):
    '''This functions converts a string timestamp into pandas datetime structure. 
    The function makes a new column by extacting the weekday.'''
    # Convert string timestamp to pandas datetime
    df[timestamp] = pd.to_datetime(df_join[timestamp])
    # Make new column with weekday
    df['day'] = df[timestamp].dt.weekday_name
```


```python
df_day(df_join, 'date')
df_join[1020:1025]  #sample of table values 
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>sales</th>
      <th>lat</th>
      <th>lon</th>
      <th>delivery time</th>
      <th>avg income</th>
      <th>zipcode</th>
      <th>city</th>
      <th>day</th>
    </tr>
    <tr>
      <th>orderid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1553535</th>
      <td>NaT</td>
      <td>NaN</td>
      <td>34.012453</td>
      <td>-118.457838</td>
      <td>1077.0</td>
      <td>39765.0</td>
      <td>90405</td>
      <td>Santa Monica</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1553536</th>
      <td>2017-08-08 02:26:01.213</td>
      <td>43.73</td>
      <td>37.779351</td>
      <td>-122.497121</td>
      <td>235.0</td>
      <td>29058.0</td>
      <td>94121</td>
      <td>San Francisco</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>1553537</th>
      <td>2017-08-08 02:30:57.383</td>
      <td>137.47</td>
      <td>37.807403</td>
      <td>-122.301621</td>
      <td>172.0</td>
      <td>14127.0</td>
      <td>94607</td>
      <td>Oakland</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>1553538</th>
      <td>2017-08-08 02:38:09.680</td>
      <td>56.94</td>
      <td>37.748804</td>
      <td>-122.423628</td>
      <td>20.0</td>
      <td>27502.0</td>
      <td>94110</td>
      <td>San Francisco</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>1553539</th>
      <td>2017-08-08 03:30:50.513</td>
      <td>65.22</td>
      <td>37.485039</td>
      <td>-122.192457</td>
      <td>119.0</td>
      <td>14781.0</td>
      <td>94063</td>
      <td>Redwood City</td>
      <td>Tuesday</td>
    </tr>
  </tbody>
</table>
</div>



A list of the features for the table above : <br>
- **orderid**: unique identifier per order (int)
- **date**: Timestamp of delivery 
- **sales**: Total product sold (USD)
- **lat**: Latitude: delivery location (degrees)
- **lon**: Longitude: delivery location (degrees)
- **delivery time**: An estimate of how long a delivery took (seconds)
- **avg income**: Average income for given zipcode  
- **zipcode**: The zipcode corresponding to delivery location  
- **city**: The city corresponding to geoinformation 
- **day**: The day of the week for delivery


```python
# Count the instances for each weekday
df_join.day.value_counts()
```




    Tuesday    3352
    Name: day, dtype: int64



## Cluster data into at least 3 and at most 20 groups

A list of the features for the clustered table below : <br>
- **city**: The city corresponding to geoinformation 
- **zipcode**: The zipcode corresponding to delivery location  
- **sales**: Average sale for zipcode  (USD)
- **delivery time**: Average delivery time for zipcode  
- **avg income**: Average income for zipcode


```python
# Average sale and income per zipcode, only showing the first 20 rows 
(df_join.groupby(['city', 'zipcode'])
 .aggregate('mean')
 .drop(df_join[['lat', 'lon']], axis=1)
 .round(1))[0:20]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>sales</th>
      <th>delivery time</th>
      <th>avg income</th>
    </tr>
    <tr>
      <th>city</th>
      <th>zipcode</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Alameda</th>
      <th>94501</th>
      <td>70.4</td>
      <td>296.7</td>
      <td>24503.0</td>
    </tr>
    <tr>
      <th>94502</th>
      <td>82.5</td>
      <td>127.0</td>
      <td>37282.0</td>
    </tr>
    <tr>
      <th>Albany</th>
      <th>94706</th>
      <td>75.7</td>
      <td>305.8</td>
      <td>24630.0</td>
    </tr>
    <tr>
      <th>Anaheim</th>
      <th>92801</th>
      <td>84.8</td>
      <td>23.0</td>
      <td>12744.0</td>
    </tr>
    <tr>
      <th>Aptos</th>
      <th>95003</th>
      <td>126.7</td>
      <td>344.0</td>
      <td>24796.0</td>
    </tr>
    <tr>
      <th>Atherton</th>
      <th>94027</th>
      <td>78.3</td>
      <td>226.6</td>
      <td>130531.0</td>
    </tr>
    <tr>
      <th>Belmont</th>
      <th>94002</th>
      <td>56.5</td>
      <td>181.4</td>
      <td>41416.0</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Berkeley</th>
      <th>94702</th>
      <td>71.5</td>
      <td>235.2</td>
      <td>21607.0</td>
    </tr>
    <tr>
      <th>94703</th>
      <td>67.2</td>
      <td>182.0</td>
      <td>19617.0</td>
    </tr>
    <tr>
      <th>94704</th>
      <td>63.6</td>
      <td>218.0</td>
      <td>6931.0</td>
    </tr>
    <tr>
      <th>94705</th>
      <td>43.8</td>
      <td>188.1</td>
      <td>42749.0</td>
    </tr>
    <tr>
      <th>94707</th>
      <td>102.5</td>
      <td>137.0</td>
      <td>40213.0</td>
    </tr>
    <tr>
      <th>94709</th>
      <td>67.1</td>
      <td>242.2</td>
      <td>18362.0</td>
    </tr>
    <tr>
      <th>94710</th>
      <td>69.7</td>
      <td>148.6</td>
      <td>18683.0</td>
    </tr>
    <tr>
      <th>Brisbane</th>
      <th>94005</th>
      <td>62.5</td>
      <td>175.1</td>
      <td>46645.0</td>
    </tr>
    <tr>
      <th>Burlingame</th>
      <th>94010</th>
      <td>52.2</td>
      <td>214.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Campbell</th>
      <th>95008</th>
      <td>62.7</td>
      <td>0.0</td>
      <td>30810.0</td>
    </tr>
    <tr>
      <th>Capitola</th>
      <th>95010</th>
      <td>92.8</td>
      <td>321.6</td>
      <td>22951.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Carson</th>
      <th>90745</th>
      <td>36.0</td>
      <td>145.0</td>
      <td>16970.0</td>
    </tr>
    <tr>
      <th>90746</th>
      <td>54.2</td>
      <td>675.5</td>
      <td>19531.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The following dataframe groups df_join by city
df_city_group = (df_join.set_index('city').groupby(level=0)['sales']
                 .agg({'avg sale': np.mean, 'sum': np.sum})
                 .astype(int))
```


```python
# Sort by sum of purchases for each city in descending order, displaying first 5 entries 
df_city_group.sort_values(by='sum', ascending=False).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg sale</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>San Francisco</th>
      <td>71</td>
      <td>72295</td>
    </tr>
    <tr>
      <th>Oakland</th>
      <td>63</td>
      <td>21071</td>
    </tr>
    <tr>
      <th>San Diego</th>
      <td>70</td>
      <td>16746</td>
    </tr>
    <tr>
      <th>San Mateo</th>
      <td>69</td>
      <td>10443</td>
    </tr>
    <tr>
      <th>Santa Cruz</th>
      <td>85</td>
      <td>8987</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sort by avg purchase for each city in descending order, displaying first 5 entries 
df_city_group.sort_values(by='avg sale', ascending=False).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg sale</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Laguna Hills</th>
      <td>126</td>
      <td>252</td>
    </tr>
    <tr>
      <th>Aptos</th>
      <td>126</td>
      <td>3548</td>
    </tr>
    <tr>
      <th>Santa Ana</th>
      <td>113</td>
      <td>226</td>
    </tr>
    <tr>
      <th>Costa Mesa</th>
      <td>113</td>
      <td>1021</td>
    </tr>
    <tr>
      <th>San Ysidro</th>
      <td>106</td>
      <td>106</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The following dataframe groups df_join by city and yields average wait time, displaying first 5 entries 
df_city_group_time = (df_join.set_index('city').groupby(level=0)['delivery time']
                 .agg({'avg time': np.mean, 'max': np.max})
                 .astype(int))
df_city_group_time.sort_values(by='avg time', ascending=False).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg time</th>
      <th>max</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Los Gatos</th>
      <td>803</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>Carson</th>
      <td>498</td>
      <td>2264</td>
    </tr>
    <tr>
      <th>Hermosa Beach</th>
      <td>452</td>
      <td>2327</td>
    </tr>
    <tr>
      <th>National City</th>
      <td>428</td>
      <td>1288</td>
    </tr>
    <tr>
      <th>Redondo Beach</th>
      <td>403</td>
      <td>1766</td>
    </tr>
  </tbody>
</table>
</div>



## Data Visualization 


```python
df_join.city.value_counts().plot(kind='bar', alpha=.5, color='orange',  figsize=(16, 8), title ='Histogram of sales by City'); 
```


![png](output_24_0.png)



```python
df_city_group['sum'].sort_values(ascending = False).plot(kind='bar', alpha=.5, color='orange',  figsize=(16, 8), title ='Total Sales per City'); 
```


![png](output_25_0.png)



```python
# What percentage of total sales do San Francisco and Oakland contribute?
def perc_sales(df, cities, sum_):
    '''A function to calculate total percentage of sales from a list of cities. 
    The inputs are a dataframe, list of cities, and a string column name for sum of sales'''
    return 'The total sales contributed for listed cities: {:0.1f}%'.format(sum([df.loc[item, sum_] for item in cities]) / df[sum_].sum() * 100)


perc_sales(df_city_group, ['San Francisco', 'Oakland'], 'sum')
```




    'The total sales contributed for listed cities: 41.1%'




```python
df_join.sales.plot.hist(
    alpha=.5, color='orange', bins=100,  figsize=(16, 8), title='Sales [$] histogram')

'Average sale ${} with a median of ${}'.format(
    "%.0f" % df_join.sales.mean(), "%.0f" % df_join.sales.median())
```




    'Average sale $69 with a median of $55'




![png](output_27_1.png)



```python
(df_city_group_time['avg time'].sort_values(ascending = False)
    .plot(kind='bar', alpha=.5, color='orange',  figsize=(16, 8), title ='Avg delivery time per City')); 
```


![png](output_28_0.png)



```python
df_join['delivery time'].plot.hist(
    alpha=.5, color='orange', bins=100,  figsize=(16, 8), title='Estimated Delivery Time [s] histogram')

'Average estimated delivery time: {} s with a median of {} s'.format(
    "%.0f" % df_join['delivery time'].mean(), "%.0f" % df_join['delivery time'].median())
```




    'Average estimated delivery time: 227 s with a median of 150 s'




![png](output_29_1.png)



```python
# plots average time versus sum of sales for city 
# merges dataframes which have total sales and average delivery, merges on cities 
df_time_sales = pd.merge(df_city_group, df_city_group_time, how='outer',
                   left_index=True, right_index=True)

sns.set(style="ticks");
x = df_time_sales['sum']
y = df_time_sales['avg time']
g = sns.jointplot(x, y, kind="hex",  color="#d95f0e")
g.fig.set_size_inches(16,8)
g.fig.suptitle('Average delivery time versus sum of sales');
```


![png](output_30_0.png)



```python
# The following code generates a heatmap with location data
# load a Numpy array of (latitude, longitude) pairs from dataframe
locations = df_join[["lat", "lon"]].dropna()
# Center map on bakersfield
bakersfield_coord = (35.376175, -119.022102)
fig = gmaps.figure(center=bakersfield_coord, zoom_level=6)
# Generate heatmap for locations
fig.add_layer(gmaps.heatmap_layer(locations))
# Used to export map as html
embed_minimal_html('map.html', views=[fig])
```

# Conclusion

## Insights to advance a company financially

Starting with the first graph above, the **Histogram of Sales by City** we see that San Francisco yields the most sales. The sales in San Francisco are nearly 3 times the sales of Oakland—the second leader in sales transactions. One could conclude that whatever marketing strategies are being used in used in San Francisco should be executed in other cities as well, especially cities in the east bay. However, local competition could be capturing the Oakland market.

The next graphic **Total Sales per City** essentially mirrors the previous. Ostensibly the number of transactions is closely correlated with total sales generated. We see San Francisco generated more than 3 times the sales of Oakland. The two cities (San Francisco and Oakland) accounted for 41% of total sales.

The next graphic of interest is the **Estimated Delivery Time [s] histogram**. It demonstrates that on the whole delivery times are well behaved with not many outlier points that demand attention for curtailing delivery times. This is great! Whichever best practices are being used here should be applauded. From the graphic **Avg delivery time per City**, we see the city of Los Gatos had the highest average delivery time (800 s), nearly the 3.5 times the average time of 227 s. The long times could be due to the few sales in that city. Nearby San Jose had average delivery times, but much greater sales than Los Gatos. From the graph **Average delivery time versus sum of sales**  we see average delivery times do not scale down with total sales, i.e. no correlation. For delivery times there seems to be a regression to the mean of 227 s. 

Unfortunately, there was insufficient day and time data. Such data could have yielded the day and time most purchases are made. Armed with that information, one such testable hypothesis, perhaps on a smaller market, could have been to run promotions on days when sales are low. 

The heatmap below shows hot spots for where most of the sales are geographical generated. Most of the high-density locations (Bay Area, Los Angeles, San Diego) are well represented. However, medium-sized cities like Sacramento and Fresno are virtually untapped. Moreover, the San Joaquin Valley is a largely untapped market.



```python
%%HTML
<iframe width = "100%" height = "475" src="map.html"></ iframe>
```


<iframe width = "100%" height = "475" src="map.html"></ iframe>

