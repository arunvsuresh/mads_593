#!/usr/bin/env python
# coding: utf-8

# # Data Merging/Index Creation
# by Arun Suresh, Nick Wheatley and James Conner

# <hr style="border:2px solid gray">

# ##### This notebook highlights how we merged our main data sources (Zillow, ACS, Equifax) and created the indexes that formed the core of our affordability index, used for measuring housing affordability nation-wide and regionally in NYC and SF. 

# <hr style="border:1px solid gray">

# ## Additional Notebooks
# 
# **[Data Manipulation](01_Data_Manipulation.ipynb)**
# 
# **[Data Analysis](03_Analysis.ipynb)**

# <hr style="border:1px solid gray">

# ### Set Up the Environment
# 
# Ensure that the environment has consistent versions of the required libraries.

# In[1]:


import sys
# Plotly is required for the graphs
# Kaleido is required to export graph images
# Pandas required for dataframes
# Numpy required for pandas
# Pyarrow installed to write/read parquet files
# Pip upgrade required due to pandas wheel build issues

# required_libs = ['pandas==1.2.5', 'numpy==1.23.4', 'plotly==5.10.0', 'pyarrow==9.0.0', 'kaleido==0.2.1']
# required_libs_str = " ".join(required_libs)

# current_libs = get_ipython().getoutput('{sys.executable} -m pip freeze')
                    
# if len(set(required_libs) - set(current_libs)) != 0:
#     print("Missing or different libraries.")
#     print("Installing required libraries/versions.")
#     get_ipython().system('{sys.executable} -m pip install --upgrade pip -q')
#     get_ipython().system('{sys.executable} -m pip install -q {required_libs_str}')
# else:
#     print("Libraries and versions match.")


# ### Import Libraries/Modules

# In[2]:


import os
import pandas as pd
import numpy as np
from collections import defaultdict


# ### Load Data

# Read in all necessary files. Here, we also read in a FIPS dataset, which contains county-level identifiers. We will use these FIPS codes further down in our analysis as well as our visuals.

# In[3]:


acs = pd.read_parquet('data/acs_5_year.parquet')
equifax = pd.read_parquet('data/equifax.parquet')
zillow = pd.read_parquet('data/zillow.parquet')
fips = pd.read_parquet('data/fips.parquet')


# <hr style="border:1px solid gray">

# ### Merging and Index Creation

# Let's fix our FIPS codes which identify each county in the US

# In[4]:


# fips data contains county level codes to distinguish each county from each other
fips.columns = ['State', 'County', 'StateCodeFIPS', 'MunicipalCodeFIPS', 'FIPS']

#prepend 0's to fips to offset earlier truncation since the read-in truncates leading 0's
fips['StateCodeFIPS'] = fips['StateCodeFIPS'].astype('str').str.zfill(2)
fips['MunicipalCodeFIPS'] = fips['MunicipalCodeFIPS'].astype('str').str.zfill(3)
fips['FIPS'] = fips['FIPS'].astype('str').str.zfill(5)


# ##### Merge datasets

# Each data source contains State and County fields which we are using as our main fields to merge on. This will align all measurable indicators such as Home Value and Income along State and County lines. We will use this final dataset to create our National and Regional (SF & NYC) narrative. Initially, we merge our Equifax and ACS data on County, State, Year, and Median Household Income using an "outer" merge. From there, we merge on the Zillow dataset on County, State, and Year using an "outer" merge. Finally, we merge on our FIPS dataset using a "left" merge. We use an outer merge for our main datasets as they each contain different columns and we want to get a holistic picture of our merged dataset. 

# In[5]:


# Merge equifax, acs and zillow along County, State, Year 
equifax_acs_merge = \
acs.merge(equifax, how='outer', \
on=['County', 'State', 'Year', 'Median Household Income'])

df = equifax_acs_merge.merge(zillow, how='outer', on=['County', 'State', 'Year'])
df.columns = [x.strip() for x in df.columns]

# Merge FIPS into the dataframe
col_list = ['Fips','MunicipalCodeFIPS', 'StateCodeFIPS']
df = df.drop(col_list, axis=1).sort_values(['State','County','Year']).merge(right=fips, on=['State','County'], how='left')


# Let's inspect the data for completeness

# In[6]:


# Check FIPS codes within final df for completeness
df[df.FIPS.isna()][['State','County']].drop_duplicates()


# In[7]:


# Median Home Value Nan pcts are minimal per year. 
# We are not using 2027 since 2027 doesn't have median home values.
df[~df.Year.isin([2027])].groupby(['Year'])['Median Home Value'].apply(lambda x: x.isna().sum() / x.count())


# In[8]:


# Total Utility Costs Nan pcts are minimal per year. 
# Using only 2022 and 2027 since those are the only years we have data for
df[df.Year.isin([2022, 2027])].groupby(['Year'])['Total Utility Costs'].apply(lambda x: x.isna().sum() / x.count())


# ##### Transform Utilities data

# As shown above, we have no utility cost data from equifax for years besides 2022 and their 2027 estimate. We will use the estimated growth to backwards interpolate data from years 2017-2022, assuming the same growth from 2022-2027, with even increments each year.

# In[9]:


# Fill missing utility data for 2017-2021 by backwards interpolating Equifax's 2027 growth 
group_cols = ['State','County']
aggregate_var = 'Total Utility Costs'

df_utility_2022 = df.loc[df.Year == 2022,group_cols + [aggregate_var]].rename(columns={
    aggregate_var:'utility_2022'})

df_utility_2027 = df.loc[df.Year == 2027,group_cols + [aggregate_var]].rename(columns={
    aggregate_var:'utility_2027'})

# Calculate the increase from 2022-2027, and apply that difference to 2022 to estimate 2017 costs
df_utility = df_utility_2022.merge(df_utility_2027,on=group_cols,how='outer')
df_utility['utility_difference'] = df_utility.utility_2027 - df_utility.utility_2022
df_utility['utility_2017'] = df_utility.utility_2022 - df_utility.utility_difference
df_utility['utility_increment'] = df_utility.utility_difference / 5

# Calculate utility costs for each year from 2018-2022 using an even increment calculated above
i = 1
for year in [2018,2019,2020,2021]:
    df_utility[f'utility_{year}'] = df_utility.utility_2017 + df_utility.utility_increment * i
    i+=1

df_utility.drop(['utility_difference','utility_increment','utility_2027'],axis=1,inplace=True)
df_utility.head()


# Let's melt the data for easy comparison along year, county, state lines

# In[10]:


# Convert utility dataset to tabular form
value_cols = sorted(df_utility.columns[~df_utility.columns.isin(group_cols)].tolist())

utility = df_utility.melt(id_vars=group_cols,value_vars=value_cols,var_name='Year',
    value_name='Total Utility Costs')
utility.Year = utility.Year.apply(lambda x: int(x[-4:]))
utility.head()


# In[11]:


# Merge utility dataset with income and median home value data
df1 = df.loc[~df.Year.isin([2016,2027]),['State','County','Year','Median Household Income',
    'Median Home Value', 'FIPS']].merge(utility,on=group_cols+['Year'],how='outer')
df1.groupby('Year').mean()


# ## Create Affordability Indexes
# 
# With the data extracted and cleaned, we can now create the indexes key to our analysis. To better understand how affordability is changing each year, we determined a relative index across the US would best display the regional shift in home prices, income, and utility costs.

# <hr style="border:1px solid gray">

# #### Sub-Indexes
# The main index we will use is an **affordability index** - a composite index comprised of the following sub-indexes:
# 
# - **Income Index**
# - **Home Price Index**
# - **Utility Cost Index**
# 
# Where the Income Index uses Median Household Income, the Home Price Index uses Median Home Value, and the Utility Cost Index uses Total Utility Costs. The objective of the indices is to portray less-costly values as more affordable. Thus, a higher index relates to greater affordability.
# 
# The Income Index will use the following equation:
# &ensp;
# $$z_i = \frac{x_i - min(x)}{max(x) - min(x)}$$
# &ensp;
# 
# and the Home Price and Utility Cost indexes will use the equation below (1-index):
# 
# &ensp;
# $$z_i = 1-\frac{x_i - min(x)}{max(x) - min(x)}$$
# &ensp;
# 
# where
# 
# z is the index
# 
# i is the region
# 
# x is an index variable

# <hr style="border:1px solid gray">

# #### Composite indexes
# 
# The **Affordability Index** is a weighted average of these sub-indexes with the following weights:
# 
# - Income Index (**25%**)
# - Housing Index (**70%**)
# - Utility Index (**5%**)
# 
# We will create two versions of these indexes:
# 1. price-based
# 2. growth-based
# 
# The price-based indexes will index based on the prices in its year. The growth-based indexes will use the percent of change from the 2020-2022 period (COVID years).

# In[12]:


# Create a 2022 - 2020 percent change dataframe for key metrics

merge_columns = ['State','County','FIPS'] # For future merging

index_rename = {
    'Median Household Income':'median_income',
    'Median Home Value':'median_home_value',
    'Total Utility Costs':'utility_cost'
}

# Create dataframes for desired years
df_2020 = df1.loc[df1.Year == 2020,merge_columns + list(index_rename.keys())].rename(
    columns={k:v+'_2020' for k,v in index_rename.items()})
df_2022 = df1.loc[df1.Year == 2022,merge_columns + list(index_rename.keys())].rename(
    columns={k:v+'_2022' for k,v in index_rename.items()})

# Merge data frames
df_change = df_2022.merge(df_2020,on=merge_columns,how='outer')
df_change['Year'] = 2022
merge_columns.append('Year')

# Calculate percent growth
for kpi in list(index_rename.values()):
    column_name = kpi+'_2yr_change_perc'
    merge_columns.append(column_name) 
    df_change[column_name] = (df_change[kpi+'_2022'] / 
    df_change[kpi+'_2020']) - 1

df_change.head()


# ###### Modularize helper functions for indexing and creating a composite index:

# In[13]:


# Function to create consolidated index
def create_weighted_index(data,composite_metric,base_metrics,weights):
    """
    Creates a composite index calculated using a weighted average from passed
    metrics and weights. 

    data: pandas DataFrame
    composite_metric: string object indicating desired composite metric name
    base_metrics: list object indicating the names of the desired weight variables
    weights: dictionary object indicating the desired weights for weight variables with metric names as keys

    returns: pandas DataFrame containing the new composite index
    """
    data[composite_metric] = \
        data[base_metrics[0]] * weights[base_metrics[0]] + \
        data[base_metrics[1]] * weights[base_metrics[1]] + \
        data[base_metrics[2]] * weights[base_metrics[2]]

    return data


# In[14]:


# Function to create index
def create_index(data,idx,indices,inverse):
    """
    Creates an index using the formulas described in the markdown above.

    data: pandas DataFrame
    idx: string object indicating desired index name and the key for the indices dictionary
    indices: dictionary object with index names as keys, and index base metrics (Total Utility Cost) as values
    inverse: boolean object that determines whether to create a pure index or its inverse

    returns: pandas DataFrame containing the new index
    """
    if inverse:
        data[idx] = \
                (1-(data[indices[idx]] - min(data[indices[idx]])) / \
                (max(data[indices[idx]]) - min(data[indices[idx]]))) * 100
    else:
        data[idx] = \
                (data[indices[idx]] - min(data[indices[idx]])) / \
                (max(data[indices[idx]]) - min(data[indices[idx]])) * 100
    return data


# ###### Create indexes:

# In[15]:


# Merge percent change dataframe with main
df1b = df1.merge(df_change[merge_columns],on=merge_columns[:4],how='left')


# Create dictionary of indexes and their base metrics
indices = {
    'income_index':'Median Household Income',
    'income_growth_index':'median_income_2yr_change_perc',
    'home_price_index':'Median Home Value',
    'home_price_growth_index':'median_home_value_2yr_change_perc',
    'utility_cost_index':'Total Utility Costs',
    'utility_cost_growth_index':'utility_cost_2yr_change_perc'
}

# Create dictionary of sub-index weights for the composite "affordability" index
afi_weights = {
    'income_index':0.25,
    'utility_cost_index':0.05,
    'home_price_index':0.70,
    'income_growth_index':0.25,
    'utility_cost_growth_index':0.05,
    'home_price_growth_index':0.70
}

# Calculate individual indexes by year
years = sorted(df1b.Year.unique())
dfs = []
for y in years:
    df_tmp = df1b.loc[df1b.Year == y].copy() # Create temporary DataFrame sliced by the current year
    
    # Create sub-indexes
    i = 0
    for idx in indices.keys():
        if idx in ['income_index','income_growth_index']:
            df_tmp = create_index(df_tmp,idx,indices,inverse=False)
 
        else:
            # For cost-related indices, subtract one from the calculation to give lower costs a higher index
            df_tmp = create_index(df_tmp,idx,indices,inverse=True)

    # Create composite price-based index
    df_tmp = create_weighted_index(
        df_tmp,
        'afi_unscaled',
        ['income_index','utility_cost_index','home_price_index'],
        afi_weights
        )

    # Create composite growth-based index
    df_tmp = create_weighted_index(
        df_tmp,
        'afi_growth_unscaled',
        ['income_growth_index','utility_cost_growth_index','home_price_growth_index'],
        afi_weights
        )

    # Scale the composite indexes between 0-100 by re-indexing them
    df_tmp = create_index(df_tmp,'afi',{'afi':'afi_unscaled'},inverse=False)
    df_tmp = create_index(df_tmp,'afi_growth',{'afi_growth':'afi_growth_unscaled'},inverse=False)

    # Create a total cost index using just home price and utility costs.
    df_tmp['total_cost_index'] = \
        df_tmp.home_price_index * 0.8 + \
        df_tmp.utility_cost_index * 0.2
    
    dfs.append(df_tmp)

df2 = pd.concat(dfs)

df2.head()


# In[16]:


# Write to disk
df2.to_parquet("data/merged_dataframe.parquet")