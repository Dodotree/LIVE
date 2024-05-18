#!/usr/bin/env python
# coding: utf-8

# # PyMC3: Analysing Hotel Cancellations with Bayesian Statistical Methods

# #### Attributions
# 
# The below example uses the PyMC3 library to implement Bayesian Statistical Methods on the analysis of the hotel cancellations dataset. Please note that the below was run using Python version 3.6.9 and PyMC3 version 3.9.2.
# 
# The latest version of PyMC3 can be installed using **pip** as follows:
# 
# ```pip3 install pymc3```
# 
# However, if you encounter difficulty in executing certain functions, then please try installing the 3.9.2 version as follows:
# 
# ```pip3 install pymc3==3.9.2```
# 
# Modifications have been made where appropriate for conducting analysis on the dataset specific to this example. The output and findings in this notebook are not endorsed by the original author in any way.
# 
# The copyright and permission notices are included below in accordance with the terms of the license:
# 
# Copyright 2020 The PyMC Developers
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# The original datasets for hotel cancellations, as well as relevant research, is available here from the original authors.
# 
# * [Antonio, Almeida, Nunes, 2019. Hotel booking demand datasets](https://www.sciencedirect.com/science/article/pii/S2352340918315191)
# 
# This full solution is provided by Manning Publications and should not be used in place of your original work. Submitting this solution as your own is plagiarism and will disqualify you from earning the Manning Certificate of Completion for this liveProject Series.

# # Milestone 1

# ### 1. Import Libraries and Load Data

# In[ ]:


import arviz as az
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import random
from math import sqrt


# In[ ]:


df=pd.read_csv("h1weekly.csv")
data=np.array(df['IsCanceled'])


# In[ ]:


plt.hist(data)
plt.show()


# In[ ]:

print("MEAN")
np.mean(data)


# In[ ]:


print("STD")
np.std(data)


# ### 2. Define priors and generate 1000 samples
# 
# - [PyMC3 documentation: prior and posterior predictive checks](https://docs.pymc.io/en/stable/pymc-examples/examples/diagnostics_and_criticism/posterior_predictive.html)
# 
# - Priors = one's beliefs about mu and sigma
# 
# When selecting the priors, we cannot know the mean and standard deviation without directly calculating it from the data. As we are told that the minimum number of cancellations is 14 and the maximum is 222 - the mean prior (mu_prior) is assumed to be the average of these two values (118, rounded up to 120 for simplicity).
# 
# The prior standard deviation (sigma_prior) is set to 10, which is chosen on the basis that we expect the standard deviation to be quite small relative to the mean, i.e. we expect little deviation from the mean of 120.

# In[ ]:


mu_prior=120
sigma_prior=10

print("BEFORE LOOP")
with pm.Model() as model:
    mu = pm.Normal("mu", mu=mu_prior, sigma=sigma_prior)
    sd = pm.HalfNormal("sd", sigma=sigma_prior) # Half of normal, also uses sigma prior
    obs = pm.Normal("obs", mu=mu, sigma=sd, observed=data)
    idata = pm.sample(1000, return_inferencedata=True)
print("AFTER LOOP")

# In[ ]:


print("MODEL LOOP")
with model:
    post_pred = pm.sample_posterior_predictive(idata.posterior)
# add posterior predictive to the InferenceData
az.concat(idata, az.from_pymc3(posterior_predictive=post_pred), inplace=True)

print("AFTER MODEL LOOP")

# ### 3. Generate posterior plots

# In[ ]:


print("PLOT_TRACE")
az.plot_trace(idata);


# In[ ]:


print("SUMMARY")
az.summary(idata)
plt.show()


# In[ ]:


print("PLOT_FOREST")
az.plot_forest(idata, r_hat=True);
plt.show()


# In[ ]:

print("PLOT_POSTERIOR")
az.plot_posterior(idata);
plt.show()


# In[ ]:


fig, ax = plt.subplots()
az.plot_ppc(idata, ax=ax)
ax.axvline(data.mean(), ls="--", color="r", label="True mean")
ax.legend(fontsize=10);
plt.show()

