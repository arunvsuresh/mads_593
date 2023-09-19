import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict

# ### Data Manipulation

# #### Zillow

# Let's first start with our Zillow data. For our analysis, we used Zillow's ZHVI (Zillow Home Value Index) dataset, which provides typical home values per county going back all the way to 2000. Median Home Values composed a large part (weight) of our Affordability Index, a metric we created as a means to measure housing affordability on a per-county basis. We'll get into further detail about the indices used in our analysis in a separate notebook. 

# ##### Read files

# Each dataset was downloaded as a CSV to start. Here we also download a file containing full US state names to replace Zillow's abbreviated State names.
def read_data(filename):
    return pd.read_csv(filename) if '.csv' in filename else pd.read_parquet(filename)


# Let's inspect the head to see how the data is constructed.

# zillow.head(2)


# ##### Transforming the Data

# As you can see, our data is constructed in a wide-format, with each month-year combination being its own column, not very informative for our purposes. This makes it hard to group and aggregate and ultimately merge our data on State, County, and Year so we'll have to melt our data from wide to long. 
# 
# In addition, part of our analysis and visualizations require the use of FIPS codes to identify different counties. Because some FIPS codes have pre-pended 0's, here we pad the State FIPS and Municipal FIPS with 0's as CSV files tend to truncate them upon reading. Let's also drop non-informative columns such as *SizeRank, RegionType, and RegionID* and only work with years 2016 and beyond. 

def transform_data(df):
    # rename 'RegionName' column to 'County' for merging purposes 
    df = df.rename(columns={'RegionName': 'County'})

    # limit years to 2016 onwards to focus on 2017-2022 timeframe
    years = \
    [str(year) for year in range(2000, 2016)]

    cols_to_drop = \
    [col for col in df.columns for year in years if year in col]


    # do not drop existing zillow fips columns, 
    # but make a combine/padded column with state fips and municipal fips, 
    # add municipal fips to state fips
    # add zfill to the str to help create FIPS codes

    df['Fips'] = \
    df['StateCodeFIPS'].astype('str').str.zfill(2) + df['MunicipalCodeFIPS'].astype('str').str.zfill(3)

    # drop non-informative columns not used for analysis
    cols_to_drop\
    .extend(['SizeRank', 'RegionType', 'StateName', 'RegionID', 'Metro'])
    df = df.drop(columns=cols_to_drop)

    # convert zillow df from wide to long for easy merging & comparison of feature columns
    df = \
    df.melt(id_vars=['County', 'State', 'StateCodeFIPS', 'MunicipalCodeFIPS', 'Fips'], var_name='Year', \
    value_name='Home Value')

    df['Year'] = df['Year'].apply(lambda x: x[:4])
    df.Year = df.Year.astype('int')

    return df


# ##### Split-Apply-Combine

# Our data came in month-year combinations, e.g. 11/2021, but we only want one datapoint (median home value) per year since we are looking at yearly variations to housing affordability around the pandemic years, so let's group our Zillow columns and take the Median Home Value. Finally we convert the grouped series to a dataframe. 

# zillow_transformed_data = transform_data(zillow)

# group zillow df by all columns except for Home Value in order 
# to take median home value by county/state/year/fips codes
# and convert to new dataframe
def get_zillow_median_home_value(df):

    df = \
    df.groupby(['County', 'State', 'Year', 'StateCodeFIPS', \
    'MunicipalCodeFIPS', 'Fips'])\
    ['Home Value'].median().to_frame('Median Home Value')

    # reset index to re-align columns
    df = df.reset_index()
    return df



# Inspecting the head again, now we see each County, State, and Year combination has a unique Median Home Value associated with it.

# ##### String Manipulation

# Because the other main datasets, Equifax and ACS, contain full state names, let's replace the abbreviated state names with the full ones for Zillow (to ease with merging on State) along with other string manipulations to strip white space and fix county names. 

# In[8]:

def clean_zillow_strings(zillow, state_names):

    # use state_names df to replace abbreviated zillow state names to full state names
    state_names['state'] = state_names['state'].str.strip()
    state_names['code'] = state_names['code'].str.strip()

    zillow['State'] = \
    zillow['State'].replace(list(state_names['code']), \
    list(state_names['state']))

    zillow['State'] = zillow['State'].str.strip()

    # Apply string manipulations to fix county names and get rid of the word 'County'.
    # Louisiana and Alaska don't use 'County' to delineate municipalities but rather
    # 'Parish' and 'Borough' respectively. So let's get rid of 'Parish' and 'Borough'. 
    zillow['County'] = \
    zillow['County'].str.replace("County", "", regex=False)

    zillow['County'] = \
    zillow['County'].mask(zillow['State'] == 'Alaska', \
    zillow['County'].str.replace('Borough', ''))

    zillow['County'] = \
    zillow['County'].mask(zillow['State'] == 'Louisiana', \
    zillow['County'].str.replace('Parish', ''))

    zillow['County'] = zillow['County'].str.strip()

    # Further inspection of county names resulted in misspelled/inconsistent County names.
    # Let's fix that here.
    zillow.County = zillow.County.str.replace("De Kalb", "DeKalb")
    zillow.County = zillow.County.str.replace("Dekalb", "DeKalb")
    zillow.County = zillow.County.str.replace("La Salle", "LaSalle")
    zillow.County = zillow.County.str.replace("Dewitt", "DeWitt")
    zillow.County = zillow.County.str.replace("De Soto", "DeSoto")
    zillow.County = zillow.County.str.replace("O Brien", "O'Brien")

    return zillow


zillow = read_data('data/Zillow_All_Homes_TimeSeries_Smoothed_Seasonally_Adjusted_By_County.csv')
state_names = read_data('data/state_names.parquet')
zillow_transformed_data = transform_data(zillow)
zillow_median_home_val = get_zillow_median_home_value(zillow_transformed_data)
zillow_cleaned = clean_zillow_strings(zillow_median_home_val, state_names)

# Now, let's check for any nan values and any major outliers in the data.

# nan ratio is rather small, so dropping is fine here
zillow_cleaned.groupby(['Year'])['Median Home Value'].apply(lambda x: x.isna().sum() / x.count())

zillow_cleaned = zillow_cleaned.dropna()
zillow_cleaned.loc[:, 'Median Home Value'] = zillow_cleaned['Median Home Value'].apply(lambda x: round(x, 2))

# zillow_cleaned['Median Home Value'].describe()

# write file to parquet to maintain the dtypes and for faster read/write
zillow_cleaned.to_parquet('data/zillow.parquet')


#### Equifax

# Now, onto the Equifax data. The Equifax dataset used provides county level shelter and utilities data for the year 2022 and their respective projections for the year 2027. Utilities such as gas and electricity are provided, but for the purposes of our analysis, we will take the sum of all utilities on a county/state/year level. Total utilties form another component of our main Affordability Index metric used to measure housing affordability. 

##### Read files

# Here we're reading the Zillow data as well since we're using the Zillow counties as the base counties for our analysis, since Zillow contains the home values.

# zillow = pd.read_parquet('data/zillow.parquet')
# equifax = \
# pd.read_csv('data/Equifax_Consumer_Expenditure_Shelter_Utilities_Detail_Comparisons.csv')


# Again, let's inspect the head to see how the data is constructed. 

# In[14]:


# equifax.head()


# ##### Transforming the Data

# The Equifax data came in fairly ordered, so all that's needed are string manipulations to fix county names, summing each individual utility cost to form a "Total Utility Costs" metric and dropping uninformative columns, e.g. *Mortgage Interest*, for the purposes of our analysis. We say "uninformative" in this case, because while *Mortgage Interest* is part of overall shelter costs, we believe this metric is more subject to vary due to actions taken by the Fed and/or the US government rather than pure market forces. Mortgage rates around the pandemic timeframe (2020-2022) didn't change all that much so we assume that leaving them out of the affordability index doesn't change the overall results. We understand that this is a potential limitation and assumption in our analysis, and we highlight that in our report as a proposed future consideration/next steps. 

# In[15]:
def clean_equifax(equifax, zillow):

    # list unique zillow counties to filter Equifax data on
    counties_to_include = list(zillow['County'].unique())
    # drop extra mortgage interest & property tax cols as well as avg household income
    equifax = equifax.drop(equifax.columns[[4, 8, 9, 11, 12]], axis=1)

    # fix spelling State names
    equifax.State.replace("Conneticut","Connecticut", inplace=True)
    equifax.State.replace("Deleware","Delaware", inplace=True)

    equifax.loc[(equifax.State == 'Louisiana') & (equifax.County == 'La Salle'), 'County'] = "LaSalle"
    equifax.County = equifax.County.str.replace("De Witt", "DeWitt")
    equifax.County = equifax.County.str.replace("De Soto", "DeSoto")

    # manipulate string columns in order to add all utilities to create total utility costs
    equifax['Median Household Income'] = \
    equifax['Median Household Income'].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype("float")

    equifax['Total Households '] = \
    equifax['Total Households '].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype("float")

    equifax['Total Shelter'] = \
    equifax['Total Shelter'].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype("float")

    equifax.iloc[:, 6:] = \
    equifax.iloc[:, 6:].applymap(lambda x: x.replace("$", "").replace(",", "")).astype('float')

    # sum all utility costs to form a Total Utility Costs field
    equifax['Total Utility Costs'] = equifax.iloc[:, 6:].sum(axis=1)
    equifax = equifax.drop(columns=['Total Shelter'])
    equifax['Total Utility Costs'] = equifax['Total Utility Costs'].apply(lambda x: round(x, 2))

    # rows 552 & 2410 contain nan values for the year 2022, so let's drop them
    equifax = equifax.drop([552, 2410])

    # fix county and state strings
    equifax['County'] = equifax['County'].str.strip()
    equifax['State'] = equifax['State'].str.strip()

    # remove the word 'County', 'Parish', 'Borough' from the end of County field name
    equifax['County'] = equifax['County'].apply(lambda x: x.replace("County", "")).str.strip()
    equifax['County'] = equifax['County'].apply(lambda x: x.replace("Parish", "")).str.strip()
    equifax['County'] = equifax['County'].apply(lambda x: x.replace("Borough", "")).str.strip()

    # filter equifax data to only include counties housed in the zillow dataset
    equifax = equifax[equifax['County'].isin(counties_to_include)]

    return equifax

# Let's inspect our 2022 for any nan or outliers. Since 2027 data only provides projections of utilities costs and not data on household income, we have a high percentage of nans as shown below. But we'll leave those nans, since we will use our 2027 utilties data further down to backwards interpolate 2022-2017 utilities data.

# In[16]:
zillow = read_data('data/zillow.parquet')
equifax = read_data('data/Equifax_Consumer_Expenditure_Shelter_Utilities_Detail_Comparisons.csv')

equifax_cleaned = clean_equifax(equifax, zillow)

# make sure 2022 data is clean. 
print('2022 nans')
print(equifax_cleaned.loc[equifax_cleaned['Year'] == 2022].isna().sum() / equifax_cleaned.shape[0])
print('2027 nans')
print(equifax_cleaned.loc[equifax_cleaned['Year'] == 2027].isna().sum() / equifax_cleaned.shape[0])


# In[17]:


print(equifax_cleaned.loc[equifax_cleaned['Year'] == 2022, ['Median Household Income', 'Total Utility Costs']].describe())


# In[18]:


# write out equifax data to parquet to maintain dtypes and for faster read/write
equifax_filepath = 'data/equifax.parquet'
equifax_cleaned.to_parquet(equifax_filepath)


# #### ACS

# Finally, onto the ACS data. ACS (American Community Survey) provides census data such as household income by demographic and age on a county level. The income field used formed the last main part of our affordability index metric, the main metric measuring housing affordability in our analysis. 

# ##### Setup file list for concatentation and build helper functions to concat and clean data

# In[19]:


year_files = \
['ACSST5Y2016.csv', 'ACSST5Y2017.csv', 'ACSST5Y2018.csv', 'ACSST5Y2019.csv', 'ACSST5Y2020.csv','ACSST5Y2021.csv']


# Let's build up our ACS data. We've constructed our file list above, which contains all the individual CSV files representing each year we're measuring. Below we'll create a helper function to concat all the ACS year files (each year from 2016 to 2021) into one dataset

# In[20]:


def append_acs_data(year_files):
    '''
    Loops through each CSV year file and reads it into a dataframe
    Concatenates each dataframe into a final df

    year_files: list containing each CSV year file
    returns: final, concatenated dataframe
    '''

    # Create an empty list to store DataFrames for each year
    acs_year_dfs = []

    # Define a mapping between year strings and corresponding integer values
    year_mapping = {'2016': 2016, '2017': 2017, '2018': 2018, '2019': 2019, '2020': 2020, '2021': 2021}

    for year in year_files:
        # Check if the year string is in the file name
        for year_str, year_int in year_mapping.items():
            if year_str in year:
                # Read the CSV file and add the 'Year' column
                acs_year_df = pd.read_csv('data/' + year, keep_default_na=False)
                acs_year_df['Year'] = year_int
                # Append the DataFrame to the list
                acs_year_dfs.append(acs_year_df)
                break  # Stop searching for year string once found

    # Check if any DataFrames were loaded
    if not acs_year_dfs:
        return print("No data loaded")

    # Concatenate all year DataFrames into one
    final_df = pd.concat(acs_year_dfs, ignore_index=True)

    return final_df



# Now, let's create a function to clean the concatenated dataset by only keeping Median Income, State & County. We'll melt our data from wide to long as each column represented a different county (not helpful when trying to merge by State/County/Year) and clean our County and State strings.

# In[21]:


def clean_acs_data(df):

    '''
        Filters concatenated df by only selecting Median Income, Year, 
        and Label (Grouping) column containing Households
        
        Cleans State and County strings 

        df: final concatenated df
        returns: cleaned, final concatenated df
    '''
    
    df['Label (Grouping)'] = df['Label (Grouping)'].str.strip()

    cols = [col for col in df.columns if '!!Median income (dollars)!!Estimate' in col or 'Label (Grouping)' in col or 'Year' in col]
    df = df[cols]

    df_median_household_income = df[df['Label (Grouping)'] == 'Households']

    df_median_household_income = \
    df_median_household_income.melt(id_vars=['Label (Grouping)', 'Year'], var_name='County', value_name='Median Household Income')

    df_median_household_income['County'] = \
    df_median_household_income['County'].str.replace("!!Median income (dollars)!!Estimate", "", regex=False)

    df_median_household_income['State'] = \
    df_median_household_income['County'].str.split(",").apply(lambda x: x[1].strip())

    df_median_household_income['County'] = \
    df_median_household_income['County'].str.split(",").apply(lambda x: x[0].strip()).str.replace("County", "", regex=False).str.strip()

    df_median_household_income['Median Household Income'] = \
    df_median_household_income['Median Household Income'].str.replace(",", "").str.extract("([0-9]+)").astype('float')

    return df_median_household_income


# Below we'll handle the fixing of specific County names such as getting rid of  'Borough' and 'Parish' from Alaska and Louisiana respectively.

# In[22]:


acs = clean_acs_data(append_acs_data(year_files))

def clean_county_names(acs, zillow):

    # list unique zillow counties to filter ACS data on
    counties_to_include = list(zillow['County'].unique())

    acs['County'] = \
    acs['County'].mask(acs['State'] == 'Alaska', acs['County'].str.replace('Borough', '')).str.strip()

    acs['County'] = \
    acs['County'].mask(acs['State'] == 'Alaska', acs['County'].str.replace('Municipality', '')).str.strip()

    acs['County'] = \
    acs['County'].mask(acs['State'] == 'Louisiana', acs['County'].str.replace('Parish', '')).str.strip()

    acs['County'] = acs['County'].str.replace("De Witt", "DeWitt")
    acs['County'] = acs['County'].str.replace("De Soto", "DeSoto")

    # filter our final ACS dataset to only include counties within the Zillow dataset
    acs = acs[acs['County'].isin(counties_to_include)]

    # drop the label grouping field as the Median Income field contains the household income value
    acs = acs.drop(columns=['Label (Grouping)'])

    return acs

cleaned_acs = clean_county_names(acs, zillow)

# Let's inspect our ACS data for any nans or outliers

# In[23]:


print('ACS nans')
print(cleaned_acs[cleaned_acs.isna().any(axis=1)].shape[0] / cleaned_acs.shape[0])
print(cleaned_acs.describe())


# The nan percentage is very small, so let's drop nan's here. We'll also write our file out to parquet.

# In[24]:


cleaned_acs = cleaned_acs.dropna()

acs_filepath = 'data/acs_5_year.parquet'
cleaned_acs.to_parquet(acs_filepath)


