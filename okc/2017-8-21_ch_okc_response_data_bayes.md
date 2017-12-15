
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc" style="margin-top: 1em;"><ul class="toc-item"><li><span><a href="#Introduction:-Online-Dating-Data" data-toc-modified-id="Introduction:-Online-Dating-Data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction: Online Dating Data</a></span></li><li><span><a href="#Gathering-Data" data-toc-modified-id="Gathering-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Gathering Data</a></span></li><li><span><a href="#Preparing-that-data" data-toc-modified-id="Preparing-that-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Preparing that data</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Visualize-messages-to-Groups" data-toc-modified-id="Visualize-messages-to-Groups-3.0.1"><span class="toc-item-num">3.0.1&nbsp;&nbsp;</span>Visualize messages to Groups</a></span></li><li><span><a href="#Messages-to-all-groups" data-toc-modified-id="Messages-to-all-groups-3.0.2"><span class="toc-item-num">3.0.2&nbsp;&nbsp;</span>Messages to all groups</a></span><ul class="toc-item"><li><span><a href="#Function-to-plot-outcomes-for-individual-goups" data-toc-modified-id="Function-to-plot-outcomes-for-individual-goups-3.0.2.1"><span class="toc-item-num">3.0.2.1&nbsp;&nbsp;</span>Function to plot outcomes for individual goups</a></span></li></ul></li><li><span><a href="#Bias" data-toc-modified-id="Bias-3.0.3"><span class="toc-item-num">3.0.3&nbsp;&nbsp;</span>Bias</a></span></li><li><span><a href="#Histogram-of-Match-%-and-messaged" data-toc-modified-id="Histogram-of-Match-%-and-messaged-3.0.4"><span class="toc-item-num">3.0.4&nbsp;&nbsp;</span>Histogram of Match % and messaged</a></span></li></ul></li></ul></li><li><span><a href="#Choosing-a-Model" data-toc-modified-id="Choosing-a-Model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Choosing a Model</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Response-rate-for-all-contacted-with-Bayesian-Statistics" data-toc-modified-id="Response-rate-for-all-contacted-with-Bayesian-Statistics-4.0.1"><span class="toc-item-num">4.0.1&nbsp;&nbsp;</span>Response rate for all contacted with Bayesian Statistics</a></span></li></ul></li></ul></li><li><span><a href="#Evaluating" data-toc-modified-id="Evaluating-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Evaluating</a></span></li><li><span><a href="#Hyperparameter-tuning" data-toc-modified-id="Hyperparameter-tuning-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Hyperparameter tuning</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Hierarchical-Model:--A/B-Test" data-toc-modified-id="Hierarchical-Model:--A/B-Test-6.0.1"><span class="toc-item-num">6.0.1&nbsp;&nbsp;</span>Hierarchical Model:  A/B Test</a></span></li></ul></li></ul></li><li><span><a href="#Predictions" data-toc-modified-id="Predictions-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Predictions</a></span></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Conclusion</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#In-person-Meeting-percentage" data-toc-modified-id="In-person-Meeting-percentage-8.0.1"><span class="toc-item-num">8.0.1&nbsp;&nbsp;</span>In person Meeting percentage</a></span></li></ul></li></ul></li></ul></div>

# Introduction: Online Dating Data 

According to The Pew Research Center, 15% of US adults have used online dating to meet a potential mate. However, no one is discussing the probability of meeting a mate. Is online dating a practical method to meet a potential mate? Fortunately, the dating site OkCupid (okc) has released data for message reply rates based on race. One of those demographics, a Latino male, shared his okc test data with me. The data reports response rates and categorizes them into five groups: White, Latina, Asian, Black, Unknown. This notebook analyzes the response rates through the lens of Bayesian Statistics and A/B testing for each demographic. The notebook is segmented into six sections followed by a conclusion:   
- Gathering Data 
- Preparing Data 
- Choosing a Model
- Evaluating 
- Hyperparameter Tuning 
- Predictions

Through analyses, the test account yielded a reply rate of 23% with a 95% CI (17.5%, 28.2%), which overlapped with OkCupid's data for the same demographic—23.1% for Latino males. The large credible intervals are due to small sample sizes. The okc data reveals that response rates for males, in general, is low. This notebook's data reveals that for a male of color, online dating may not be the best use of time.    

Ref: https://www.pewresearch.org/fact-tank/2016/02/29/5-facts-about-online-dating/


```python
import pandas as pd
import seaborn as sns
import pymc3 as pm
import numpy as np
import matplotlib
import warnings
warnings.filterwarnings("ignore")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
%matplotlib inline
```

# Gathering Data
The code below loads the Excel file and selects the data within the first sheet. The data was collected by browsing messages for the test account and registering replies into an Excel file


```python
x1 = pd.ExcelFile('../data/response_data.xlsx')  
x1.sheet_names
df = x1.parse("Sheet1", header = 17, usecols = [0,1,2,3,4,5,6,7,8])
df.drop('Single', inplace=True, axis=1)
df.columns
```




    Index(['Date', 'Contact', 'Length', 'Ethnicity', 'Response', 'Age', 'Meeting',
           'Match'],
          dtype='object')



A list of the data variables recorded for each message: <br>
- **Date**: The date the message was sent or received.
- **Contact**: Whether the first messaged was initiated by other, 1 for other. 
- **Length**: Number of words used in first message. 
- **Ethnicity**: Demographic of recipient or sender, as identified by okc
- **Response**: A 1 for messages that recieved replies 
- **Age**: Age of message recipient 
- **Meeting**: A 1 for interactions that lead to in person meeting
- **Match**: The Match % with recipient, as identified by okc


```python
df.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Contact</th>
      <th>Length</th>
      <th>Ethnicity</th>
      <th>Response</th>
      <th>Age</th>
      <th>Meeting</th>
      <th>Match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>249</th>
      <td>2017-08-04</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>W</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>250</th>
      <td>2017-08-05</td>
      <td>NaN</td>
      <td>20.0</td>
      <td>W</td>
      <td>1.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>251</th>
      <td>2017-08-11</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>B</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>252</th>
      <td>2017-08-16</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>A</td>
      <td>NaN</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>253</th>
      <td>2017-08-19</td>
      <td>NaN</td>
      <td>32.0</td>
      <td>W</td>
      <td>1.0</td>
      <td>32.0</td>
      <td>NaN</td>
      <td>86.0</td>
    </tr>
  </tbody>
</table>
</div>



# Preparing that data


```python
# For Ethnicity column, change from a dictinoary W to White, U and NaN to Unknown, L to Latina, A to Asian, B to Black,
group_dict = {'U': 'Unknown', 'W': 'White', 'L': 'Latina',
              'A': 'Asian', 'B': 'Black', None: 'Unknown'}
df.Ethnicity.replace(group_dict, inplace=True)

# In Response and Meeting columns, change NaN to 0.
df.Response.fillna(0, inplace=True)
df.Meeting.fillna(0, inplace=True)
df.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Contact</th>
      <th>Length</th>
      <th>Ethnicity</th>
      <th>Response</th>
      <th>Age</th>
      <th>Meeting</th>
      <th>Match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>249</th>
      <td>2017-08-04</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>White</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>250</th>
      <td>2017-08-05</td>
      <td>NaN</td>
      <td>20.0</td>
      <td>White</td>
      <td>1.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>251</th>
      <td>2017-08-11</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>Black</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>252</th>
      <td>2017-08-16</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>Asian</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>253</th>
      <td>2017-08-19</td>
      <td>NaN</td>
      <td>32.0</td>
      <td>White</td>
      <td>1.0</td>
      <td>32.0</td>
      <td>0.0</td>
      <td>86.0</td>
    </tr>
  </tbody>
</table>
</div>



### Visualize messages to Groups 
The scatter plot below doesn't appear to demonstrate any correlation between message length ("Length"), match percentage ("Match"), age ('Age'), or replies ('Response'). The closest to a trend is perhaps Match versus Age. It makes sense that the test account would have a higher match percentage with people closer to his age.

Most messages were sent to recipients with a high match percentage, as demonstrated from the middle center histogram. The red dots indicate Response = 0, or no response; Blue, received a response. There doesn't seem to be a correlation between Match % or message Length for messages that received responses or not, top middle plot. It's also interesting to note that messages sent to recipients with Match % < 40 have a 50% reply rate, bottom center plot. Yet, messages with Match % > 90 and message length < 10 have a 25% reply rate, top center plot. 

The sinister diagonal displays density plots for the categories. It should be noted that the None or NaN values are not counted in the density plots. Thus, the amplitudes are not accurate, but the shapes most likely are. 



```python
g = sns.pairplot(df, vars=["Length", "Match", 'Age'], hue="Response", palette="Set1", diag_kind="kde", size=3.5)
```


![png](output_10_0.png)


### Messages to all groups

The plot below demonstrates the total number of messages sent, and the total number of replies. 


```python
# messages sent by test
sent = df.loc[(df.Contact != 1), 'Response']
```


```python
# Plot the data
fig = plt.figure(figsize=(12, 3))
gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

ax1.plot(range(len(sent)), sent, 'ro')
ax2.hist(-sent, bins=2, alpha=.5, color='orange')

ax1.yaxis.set(ticks=(0, 1), ticklabels=('No reply', 'Reply'))
ax2.xaxis.set(ticks=(-1, 0), ticklabels=('Reply', 'No reply'))

ax1.set(title='Message outcomes', xlabel='Messages', ylim=(-0.2, 1.2))
ax2.set(ylabel='Frequency')

fig.tight_layout()
```


![png](output_14_0.png)


#### Function to plot outcomes for individual goups


```python
def plot_outcomes(group_list):
    for str in group_list:
        '''This function takes a list containing demographic names as strings. 
        It returns a plot representing the number of messages with replies and non replies.'''
        # messages sent by me to group
        group = df.loc[(df.Contact != 1) & (df.Ethnicity == str),
                       'Response']

        fig = plt.figure(figsize=(12, 2))
        gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        ax1.plot(range(len(group)), group, 'ro')
        ax2.hist(-group, bins=2, alpha=.5, color='orange')

        ax1.yaxis.set(ticks=(0, 1), ticklabels=('No reply', 'Reply'))
        ax2.xaxis.set(ticks=(-1, 0), ticklabels=('Reply', 'No reply'))

        ax1.set(title='Message outcomes to {} group'.format(
            str), xlabel='Messages', ylim=(-0.2, 1.2))
        ax2.set(ylabel='Frequency')
        fig.tight_layout()
    pass
```


```python
# Plot the replies for each demographic in the list Groups
Groups = ['White', 'Latina', 'Asian', 'Black', 'Unknown']
plot_outcomes(Groups)
```


![png](output_17_0.png)



![png](output_17_1.png)



![png](output_17_2.png)



![png](output_17_3.png)



![png](output_17_4.png)


### Bias
Did the test account favor one group over the other, or is this typical of random sampling? How does this compare to okc's overall demographic makeup? According to Quantcast (2009) okc demographics were White = 91%, Hispanic = 3%, Asian = 3%, Black = 2%. There is no reason to believe okc demographics scaled differently since 2009, and it gives us something to compare. 

To compare bias I'll use the normalized bias equation $\frac{Measured-Actual}{Measured+Actual}$. The metric ranges from -1 to 1, with 0 indicating the absence of bias. The results in the table below are interesting: Even though most messages went to the 'White' group, the normalized bias yielded a negative value. This negative value is due to the overrepresented 'White' group sample size. 



```python
def bias(group_dict, dframe):
    '''A function that takes in a group dictionary and a dataframe to calculate the bias for each group. 
    The function returns a dataframe with a bias scale -1 to 1.'''
    new_list = []
    for item in group_dict:
        # messages sent to specific group
        sent = len(dframe.loc[(dframe.Contact != 1) & (
            dframe.Ethnicity == item), 'Response'])
        # find % of messages sent to one group divided by all messages sent not including Unknown group
        perc_sent = (sent / len(dframe.loc[(dframe.Contact != 1)
                                           & (dframe.Ethnicity != 'Unknown'), 'Response'])) * 100
        # okc demo %
        perc_okc = group_dict[item]
        # build list for dataframe
        # Calulate relative group bias
        bias = (perc_sent - perc_okc) / (perc_sent + perc_okc)
        new_list.append([sent, perc_sent, perc_okc, bias])
    return pd.DataFrame(np.array(new_list).reshape(4, 4), 
                        columns=['Sent', 'Sent %', 'okc group %', 'Bias'], index=list(group_dict.keys())).round(2)
```


```python
okc_demo = {'White':91,'Latina':3,'Asian':3,'Black':2}
bias(okc_demo,df)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sent</th>
      <th>Sent %</th>
      <th>okc group %</th>
      <th>Bias</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>White</th>
      <td>60.0</td>
      <td>52.17</td>
      <td>91.0</td>
      <td>-0.27</td>
    </tr>
    <tr>
      <th>Latina</th>
      <td>32.0</td>
      <td>27.83</td>
      <td>3.0</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>Asian</th>
      <td>20.0</td>
      <td>17.39</td>
      <td>3.0</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>Black</th>
      <td>3.0</td>
      <td>2.61</td>
      <td>2.0</td>
      <td>0.13</td>
    </tr>
  </tbody>
</table>
</div>



### Histogram of Match % and messaged 

According to OkCupid, the Match % demonstrates how well one matches up against a potential partner. From the histogram below we can see the median for matches contacted was 80%. 


```python
df.Match.plot.hist(alpha = .5, color = 'orange', bins=20);
'Match % average of {} with a median of {}'.format("%.0f" % df.Match.mean(),"%.0f" % df.Match.median())
```




    'Match % average of 78 with a median of 80'




![png](output_23_1.png)


# Choosing a Model

### Response rate for all contacted with Bayesian Statistics 
The reply data is modeled with a Bernoulli distribution. The distribution is a function of the parameter theta—the reply rate. Since we don't have the test account's prior reply rates, we'll model theta prior as a Uniform distribution. The observed data is stored in 'sent' (defined in Messages to all groups), which is a dataframe column containing a 1 if a message received a response, and zero otherwise.


```python
with pm.Model():
    # prior
    theta = pm.Uniform('theta', 0, 1)

    # likelihood
    Y = pm.Bernoulli('Y', p=theta, observed=sent)

    # Inference
    trace = pm.sample(4000, tune=500, njobs=2)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using ADVI...
    Average Loss = 128.42:   4%|▎         | 7000/200000 [00:01<00:27, 6981.15it/s]
    Convergence archived at 7400
    Interrupted at 7,400 [3%]: Average Loss = 139.33
    100%|██████████| 4500/4500 [00:05<00:00, 844.37it/s]


# Evaluating


```python
pm.plots.plot_posterior(trace)
pm.traceplot(trace);
```


![png](output_28_0.png)



![png](output_28_1.png)



```python
pm.summary(trace)
```

    
    theta:
    
      Mean             SD               MC Error         95% HPD interval
      -------------------------------------------------------------------
      
      0.230            0.028            0.000            [0.175, 0.282]
    
      Posterior quantiles:
      2.5            25             50             75             97.5
      |--------------|==============|==============|--------------|
      
      0.178          0.211          0.230          0.249          0.286
    


The test account experienced an average reply rate of 23% with a 95 % CI (17.5%, 28.2%). From the chart below we see this is in agreement with okc data for a Latino male. The next step is to look at response rate per individual groups.

<img src="../figures/Reply_rate_by_sender.png">

Source: [OkCupid Blog](https://theblog.okcupid.com/how-your-race-affects-the-messages-you-get-39c68771b99e)

# Hyperparameter tuning

### Hierarchical Model:  A/B Test  
For the model above I used the Classical Bernoulli distribution and assumed a uniform prior. However, we can use a Beta distribution, with hyper parameters a and b. This custom Beta distribution returns the logit, which incorporates the mean log(a/b) and the log of the "sample size", log(a+b). This analysis incorporates ideas about the logit to get desired mathematical properties when sampling for a and b. The large credible intervals signal that the sample size is too small to be statistically significant. 


Below I build a dataframe with trials and success for each group


```python
def n_trials(group_list):
    '''This function builds a list for a dataframe. The dataframe columns will be total message replies and sent.'''
    new_array = []
    for item in group_list:
        # replies
        a = df.loc[(df.Contact != 1) & (df.Ethnicity == item), 'Response'].sum()
        # messages sent to specific group
        b = len(df.loc[(df.Contact != 1) & (df.Ethnicity == item), 'Response'])
        new_array.append([a, b])
    return pd.DataFrame(np.array(new_array).reshape(5, 2), columns=['Replies', 'Sent']).set_index([group_list])
```


```python
Groups = ['White', 'Latina', 'Asian', 'Black', 'Unknown']
df1 = (n_trials(Groups))
df1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Replies</th>
      <th>Sent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>White</th>
      <td>15.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>Latina</th>
      <td>8.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>Asian</th>
      <td>9.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Black</th>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Unknown</th>
      <td>19.0</td>
      <td>118.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pymc


@pymc.stochastic(dtype=np.float64)
def hyperpriors(value=[1.0, 1.0]):
    a, b = value[0], value[1]
    if a <= 0 or b <= 0:
        return -np.inf
    else:
        return np.log(np.power((a + b), -2.5))


a = hyperpriors[0]
b = hyperpriors[1]
```


```python
# The hidden, true rate for each group.
true_rates = pymc.Beta('true_rates', a, b, size=5)

# The observed values
trials = df1.Sent.as_matrix()
# Passes array of values
successes = df1.Replies.as_matrix()
# Passes array
observed_values = pymc.Binomial(
    'observed_values', trials, true_rates, observed=True, value=successes)

model = pymc.Model([a, b, true_rates, observed_values])
mcmc = pymc.MCMC(model)

# Generate 1M samples, and throw out the first 500k
mcmc.sample(1000000, 500000)
```

     [-----------------100%-----------------] 1000000 of 1000000 complete in 188.9 sec


```python
plt.figure(figsize=(15,8))
for i in range(5):
    sns.kdeplot(mcmc.trace('true_rates')[:][:,i], shade = True, label = Groups[i])
```


![png](output_39_0.png)



```python
# save means to list 
true_rate = []
for i in range(5):
    true_rate.append( mcmc.trace('true_rates')[:][:,i].mean() ) 

# save means to df
df1['True rate %']  = pd.DataFrame({'True rate %': true_rate}, index=df1.index).round(3)*100

```


```python
df1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Replies</th>
      <th>Sent</th>
      <th>True rate %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>White</th>
      <td>15.0</td>
      <td>60.0</td>
      <td>25.7</td>
    </tr>
    <tr>
      <th>Latina</th>
      <td>8.0</td>
      <td>32.0</td>
      <td>26.1</td>
    </tr>
    <tr>
      <th>Asian</th>
      <td>9.0</td>
      <td>20.0</td>
      <td>38.9</td>
    </tr>
    <tr>
      <th>Black</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>39.6</td>
    </tr>
    <tr>
      <th>Unknown</th>
      <td>19.0</td>
      <td>118.0</td>
      <td>17.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
mcmc.summary()
```

    
    true_rates:
     
    	Mean             SD               MC Error        95% HPD interval
    	------------------------------------------------------------------
    	0.257            0.051            0.0              [ 0.156  0.358]
    	0.261            0.068            0.002            [ 0.137  0.398]
    	0.389            0.1              0.003            [ 0.204  0.582]
    	0.396            0.167            0.005            [ 0.122  0.75 ]
    	0.177            0.035            0.001            [ 0.111  0.247]
    	
    	
    	Posterior quantiles:
    	
    	2.5             25              50              75             97.5
    	 |---------------|===============|===============|---------------|
    	0.163            0.221           0.255          0.29          0.366
    	0.142            0.214           0.257          0.305         0.404
    	0.217            0.315           0.382          0.456         0.601
    	0.149            0.271           0.364          0.493         0.792
    	0.113            0.152           0.175          0.2           0.249
    	
    
    hyperpriors:
     
    	Mean             SD               MC Error        95% HPD interval
    	------------------------------------------------------------------
    	6.26             8.475            0.488          [  0.09   21.659]
    	16.71            26.168           1.589          [  0.068  63.161]
    	
    	
    	Posterior quantiles:
    	
    	2.5             25              50              75             97.5
    	 |---------------|===============|===============|---------------|
    	0.503            1.735           3.435          7.053         30.834
    	0.752            3.424           7.796          18.098        93.869
    	



```python
HPD = [ '15.6, 35.8', '13.7, 39.8', '20.4, 58.2', '12.2, 77.5',  '11.1, 24.7' ] 
df1['95% Credible Interval']  = pd.DataFrame({'95% Credible interval': HPD}, index=df1.index)
df1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Replies</th>
      <th>Sent</th>
      <th>True rate %</th>
      <th>95% Credible Interval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>White</th>
      <td>15.0</td>
      <td>60.0</td>
      <td>25.7</td>
      <td>15.6, 35.8</td>
    </tr>
    <tr>
      <th>Latina</th>
      <td>8.0</td>
      <td>32.0</td>
      <td>26.1</td>
      <td>13.7, 39.8</td>
    </tr>
    <tr>
      <th>Asian</th>
      <td>9.0</td>
      <td>20.0</td>
      <td>38.9</td>
      <td>20.4, 58.2</td>
    </tr>
    <tr>
      <th>Black</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>39.6</td>
      <td>12.2, 77.5</td>
    </tr>
    <tr>
      <th>Unknown</th>
      <td>19.0</td>
      <td>118.0</td>
      <td>17.7</td>
      <td>11.1, 24.7</td>
    </tr>
  </tbody>
</table>
</div>



# Predictions 

Let's examine the posterior of the delta distribution below for the demographics 'Asian' and 'Latina', by subtracting the former from the latter. The orange line at x = 0, represents where the difference between the two distributions is 0. We can see the delta distribution is shifted to the right about x = 0. This means most of the points sampled from 'Asian' distribution are roughly greater to those sampled from the 'Latina' distribution, implying the 'Asian' reply rate is likely greater than 'Latina'. This can be done with any demographic permutation. 

Below the delta distribution plot, I demonstrate more quantitative results by computing the probability that 'Asian' replies more than 'Latina'.


```python
delta_distribution = mcmc.trace('true_rates')[:][:, 2] - mcmc.trace('true_rates')[:][:, 1]  # subtracts 'Asian' from 'Latina'

sns.kdeplot(delta_distribution, shade=True)
plt.axvline(0.00, color='orange')
```




    <matplotlib.lines.Line2D at 0x11ddf3eb8>




![png](output_45_1.png)



```python
print ( "Probability that 'Asian' replies MORE than 'Latina': %0.3f" % (delta_distribution > 0).mean() )
print ( "Probability that 'Asian' replies LESS than 'Latina': %0.3f" % (delta_distribution < 0).mean() )
```

    Probability that 'Asian' replies MORE than 'Latina': 0.869
    Probability that 'Asian' replies LESS than 'Latina': 0.131


# Conclusion 
The posterior of the delta distribution yields a reliable way to compare one demographics reply rate versus another. Although the reply rates for the Asian and Black demographics were high, I suspect we would see a regression to the mean with more data. 

Reply rates are informative, however, actual in-person meetings should be the key metric evaluated. The analysis below demonstrates that for approximately every 100 messages sent, two dates are probable: 1.9% with a 95% CI (0.5%, 3.5%). Is that a good rate? I don't know if it's a good rate, but it does agree with the metrics from other males who have shared their online dating experience with me. Is online-dating time-effective for meeting a mate? At this point, I would have to answer in the negative. It seems like the time invested in searching, reading, and contacting profiles would be better spent on hobbies or activities where one would interact with potential mates, i.e. old school networking.

### In person Meeting percentage 
For this model, the prior is modeled as a uniform distribution. The likelihood models the in-person 'Meeting' data as a Bernoulli distribution. 


```python
with pm.Model():
    # prior 
    theta = pm.Uniform('theta', 0, 1) 
    
    # likelihood 
    Y = pm.Bernoulli('Y', p = theta, observed = df.Meeting )

    # Inference
    trace = pm.sample(5000, tune=500)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using ADVI...
    Average Loss = 31.104:   5%|▍         | 9681/200000 [00:01<00:29, 6552.21it/s]
    Convergence archived at 9700
    Interrupted at 9,700 [4%]: Average Loss = 80.888
    100%|██████████| 5500/5500 [00:03<00:00, 1500.14it/s]



```python
pm.plots.plot_posterior(trace)
pm.traceplot(trace)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x117e51a58>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1151ed5f8>]], dtype=object)




![png](output_50_1.png)



![png](output_50_2.png)



```python
pm.summary(trace)
```

    
    theta:
    
      Mean             SD               MC Error         95% HPD interval
      -------------------------------------------------------------------
      
      0.019            0.008            0.000            [0.005, 0.035]
    
      Posterior quantiles:
      2.5            25             50             75             97.5
      |--------------|==============|==============|--------------|
      
      0.007          0.013          0.018          0.024          0.039
    

