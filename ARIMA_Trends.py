#!/usr/bin/env python
# coding: utf-8

# # ARIMA Modelling: Forecasting Weekly Hotel Cancellations

# #### Attributions
# 
# The below code uses the [pmdarima](https://github.com/alkaline-ml/pmdarima) library (Copyright (c) 2017 Taylor G Smith) in executing the below examples, as provided under the MIT License.
# 
# Modifications have been made where appropriate for conducting analysis on the dataset specific to this example. The work and findings in this notebook are not endorsed by the original author in any way.
# 
# The copyright and permission notices are made available below:
# 
# Copyright (c) 2017 Taylor G Smith
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# The original datasets for hotel cancellations, as well as relevant research, is available here from the original authors.
# 
# * [Antonio, Almeida, Nunes, 2019. Hotel booking demand datasets](https://www.sciencedirect.com/science/article/pii/S2352340918315191)
# 
# This full solution is provided by Manning Publications and should not be used in place of your original work. Submitting this solution as your own is plagiarism and will disqualify you from earning the Manning Certificate of Completion for this liveProject Series.

# # Milestone 1
# 
# The below defines the procedures for forming the weekly time series as specified.

# ### Import Libraries and define dtypes

# In[1]:


import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pmdarima as pm
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf

from numpy.random import seed
seed(1)
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.stattools as ts

dtypes = {
        'IsCanceled':                                    'float64',
        'LeadTime':                                          'float64',
        'StaysInWeekendNights':                                     'float64',
        'StaysInWeekNights':                                     'float64',
        'Adults':                            'float64',
        'Children':                            'float64',
        'Babies':                                  'float64',
        'Meal':                                    'category',
        'Country':                                               'category',
        'MarketSegment':                                    'category',
        'DistributionChannel':                                       'category',
        'IsRepeatedGuest':                               'float64',
        'PreviousCancellations':                                    'float64',
        'PreviousBookingsNotCanceled':                          'float64',
        'ReservedRoomType':                                             'category',
        'AssignedRoomType':                                            'category',
        'BookingChanges':                                                'float64',
        'DepositType':                                              'category',
        'Agent':                                              'category',
        'Company':                                 'category',
        'DaysInWaitingList':                                           'float64',
        'CustomerType':                                           'category',
        'ADR':                                          'float64',
        'RequiredCarParkingSpaces':                                      'float64',
        'TotalOfSpecialRequests':                                              'float64',
        'ReservationStatus':                                                'category'
        }


# ### Data is sorted by year and week number

# In[2]:


train_df = pd.read_csv('H1.csv', dtype=dtypes, converters={'ArrivalDateWeekNumber': '{:0>2}'.format})
a=train_df.head()
b=train_df
b
c=b.sort_values(['ArrivalDateYear','ArrivalDateWeekNumber'], ascending=True)
c=pd.DataFrame(c)
c
type(c)


# In[3]:


c


# In[4]:


df = DataFrame(c, columns= ['ArrivalDateYear', 'ArrivalDateWeekNumber']) 
df


# ### Year and month are joined together (concatenated)
# 
# Use pandas manipulation procedures in order to join the year and month as the timestamp. The below guide offers a detailed explanation of how to concatenate values.
# 
# [Data to Fish: How to Concatenate Column Values in Pandas DataFrame](https://datatofish.com/concatenate-values-python/)

# In[5]:


df1 = df['ArrivalDateYear'].map(str) + df['ArrivalDateWeekNumber'].map(str)
print (df1)
df1=pd.DataFrame(df1)


# ### Cancellation entries are joined with the associated date

# In[6]:


df2 = DataFrame(c, columns= ['IsCanceled']) 
df2


# In[7]:


type(df1)


# In[8]:


df3=pd.concat([df1, df2], axis = 1)
df3
df3.columns = ['FullDate', 'IsCanceled']


# In[9]:


df3
df3.sort_values(['FullDate','IsCanceled'], ascending=True)


# ### Cancellations are aggregated on a weekly basis

# In[10]:


df4 = df3.groupby('FullDate').agg(sum)
df4
df4.sort_values(['FullDate'], ascending=True)


# In[11]:


tseries=df4['IsCanceled']
tseries


# In[12]:


#plt.plot(tseries)
#plt.tick_params(
#    axis='x',
#    which='both',
#    bottom=False,
#    top=False,
#    labelbottom=False)
#plt.show()


# Note the components of each time series as graphed below. This provides important visual signals for the structure of the time series, and is important for a more intuitive understanding of the eventual ARIMA model structure that will be chosen.

# # Milestone 2
# 
# The time series is decomposed, autocorrelation and partial autocorrelation plots are generated, a 4-week moving average is calculated, and a 90/10 train-test split is conducted.

# ### Time Series Decomposition

# In[13]:

decomposition = seasonal_decompose(tseries, period=52)
residual=decomposition.resid
trend=decomposition.trend
seasonal=decomposition.seasonal

#plt.subplot(221)
#plt.plot(tseries,color='#ff0000', label='Series')
#plt.legend(loc='best')
#plt.subplot(222)
#plt.plot(trend,color='#1100ff', label='Trend')
#plt.legend(loc='best')
#plt.tight_layout()
#plt.subplot(223)
#plt.plot(residual,color='#00ff1a', label='Residual')
#plt.legend(loc='best')
#plt.tight_layout()
#plt.subplot(224)
#plt.plot(seasonal,color='#de00ff', label='Seasonality')
#plt.legend(loc='best')
#plt.tight_layout()
#plt.show()


# In[14]:


df=pd.DataFrame(tseries)
df


# ### Autocorrelation and Partial Autocorrelation Plots
# 
# In the autocorrelation plot below, the strongest correlation after the period of negatively correlated lags occurs at t=51. In this regard, **m=51** is chosen as the seasonal parameter in the ARIMA model.
# 
# This resource provides some useful information in explaining how the seasonality cycle is determined using the autocorrelation function: [Oracle Crystal Ball Predictor User's Guide: Identifying Seasonality with Autocorrelations](https://docs.oracle.com/cd/E57185_01/CBPUG/PRHistData_Autocorr.htm)

# In[15]:


acf(tseries, nlags=100)


# In[16]:


plot_acf(df.IsCanceled, lags=100, zero=False);


# In[17]:


#pacf(tseries, nlags=100)
pacf(tseries, nlags=56)


# In[18]:


plot_pacf(df.IsCanceled, lags=10, zero=False);


# ### Moving Average

# In[19]:

window_size = 4

numbers_series = pd.Series(tseries)
windows = numbers_series.rolling(window_size)
moving_averages = windows.mean()

moving_averages_list = moving_averages.tolist()
ts4 = moving_averages_list[window_size - 1:]
ts4=np.array(ts4)
print(ts4)


# In[20]:


#n1=math.nan
#n4=np.array([n1,n1,n1,n1])
#ts4=np.concatenate([n4,ts4])
#plt.plot(tseries)
#plt.plot(ts4)
#plt.title("4-period Simple Moving Average")
#plt.show()

import matplotlib.dates as mdates

n1 = math.nan
n4 = np.array([n1, n1, n1, n1])
ts4 = np.concatenate([n4, ts4])

fig, ax = plt.subplots()
ax.plot(tseries)
ax.plot(ts4)

# Format x-axis labels as MM:DD and rotate them
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m:%d'))
plt.xticks(rotation=45)

# Reduce the number of x-axis ticks
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # One tick per week

plt.title("4-period Simple Moving Average")
plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.show()

# ### Train-Test Split

# In[21]:


tseriesr=pd.Series(tseries)
type(tseriesr)
train, test = tseriesr[1:103], tseriesr[104:115]


# In[22]:


type(test)


# In[23]:


train


# In[24]:


test

