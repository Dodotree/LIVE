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

import matplotlib.dates as mdates

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


# In[10]:


type(df3)


# ### Cancellations are aggregated on a weekly basis

# In[11]:


df4 = df3.groupby('FullDate').agg(sum)
df4
df4.sort_values(['FullDate'], ascending=True)


# In[12]:


tseries=df4['IsCanceled']
tseries


# In[13]:


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

# In[14]:


decomposition=seasonal_decompose(tseries, period = 52)
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


# In[15]:


df=pd.DataFrame(tseries)
df


# ### Autocorrelation and Partial Autocorrelation Plots
# 
# In the autocorrelation plot below, the strongest correlation after the period of negatively correlated lags occurs at t=51. In this regard, **m=51** is chosen as the seasonal parameter in the ARIMA model.
# 
# This resource provides some useful information in explaining how the seasonality cycle is determined using the autocorrelation function: [Oracle Crystal Ball Predictor User's Guide: Identifying Seasonality with Autocorrelations](https://docs.oracle.com/cd/E57185_01/CBPUG/PRHistData_Autocorr.htm)

# In[16]:


acf(tseries, nlags=100)


# In[17]:


plot_acf(df.IsCanceled, lags=100, zero=False);


# In[18]:


pacf(tseries, nlags=56)


# In[19]:


plot_pacf(df.IsCanceled, lags=10, zero=False);


# ### Moving Average

# In[20]:


window_size = 4

numbers_series = pd.Series(tseries)
windows = numbers_series.rolling(window_size)
moving_averages = windows.mean()

moving_averages_list = moving_averages.tolist()
ts4 = moving_averages_list[window_size - 1:]
ts4=np.array(ts4)
print(ts4)


# In[21]:


n1=math.nan
n4=np.array([n1,n1,n1,n1])
ts4=np.concatenate([n4,ts4])
#plt.plot(tseries)
#plt.plot(ts4)
#plt.title("4-period Simple Moving Average")
#plt.show()


# ### Train-Test Split

# In[22]:


tseriesr=pd.Series(tseries)
type(tseriesr)
train, test = tseriesr[1:103], tseriesr[104:115]


# In[23]:


type(test)


# In[24]:


train


# In[25]:


test


# # Milestone 3
# 
# **pmdarima** is used to automatically generate the coordinates for the ARIMA model based on which model shows the lowest BIC (Bayesian Information Criterion). Please refer to these two references for more information on the installation and configuration of the ARIMA model under this package.
# 
# [GitHub Repository: alkaline-ml/pmdarima](https://github.com/alkaline-ml/pmdarima)
# 
# [pypi.org: pmdarima](https://pypi.org/project/pmdarima/)

# ### auto_arima

# Looking at the autocorrelation function, the correlation drops off after 51 lags.
# 
# In this regard, the seasonal factor (m) is set to **51** in the ARIMA configuration below.

# In[26]:


Arima_model=pm.auto_arima(train, start_p=0, start_q=0, max_p=10, max_q=10, start_P=0, start_Q=0, max_P=10, max_Q=10, m=51, stepwise=True, seasonal=True, information_criterion='bic', trace=True, d=1, D=1, error_action='warn', suppress_warnings=True, random_state = 20, n_fits=30)


# In[27]:


Arima_model.summary()


# In[28]:


prediction=pd.DataFrame(Arima_model.predict(n_periods=11), index=test.index)
prediction.columns = ['Predicted_Cancellations']
predictions=prediction['Predicted_Cancellations']


# In[29]:


prediction


# In[30]:


test


# In[31]:

#plt.figure(figsize=(15,10))
#plt.plot(train, label='Training')
#plt.plot(test, label='Validation')
#plt.plot(prediction, label='Predicted')
#plt.legend(loc = 'upper center')
#plt.show()


import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(15,10))

ax.plot(train, label='Training')
ax.plot(test, label='Validation')
ax.plot(prediction, label='Predicted')

# Format x-axis labels as MM:DD and rotate them
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m:%d'))
plt.xticks(rotation=45)

# Reduce the number of x-axis ticks
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # One tick per week

ax.legend(loc = 'upper center')
plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.show()





# ### Calculate Test RMSE

# In[32]:

# Remove NaN values
#test = test[~np.isnan(test)]
#predictions = predictions[~np.isnan(predictions)]

# Remove NaN values
#test = test.dropna()
#predictions = predictions.dropna()

mse = mean_squared_error(test, predictions)

rmse = math.sqrt(mse)
print('RMSE: %f' % rmse)


# In[33]:


np.mean(test)


# ### Interpretation

# The most suitable ARIMA model under this circumstance was identified as **ARIMA(2,1,0)(0,1,0)[51]**, with a BIC of **561.339**.
# 
# The RMSE of 31 is 21% the size of the mean weekly cancellations of 146 for the test set. While a lower RMSE would be preferable, too low an RMSE value would potentially indicate that the model has been "over-trained" on the training set and will do well in forecasting the test set in question, but may perform poorly when it comes to forecasting subsequent data. Therefore, while RMSE is important, it is not regarded as the "be all and end all" when it comes to making accurate forecasts.
# 
# Given that a visual scan of the predictions indicate that the model is capturing the shift in trends across different periods, this is an indication that the model is working well for these purposes. 

# In[ ]:




