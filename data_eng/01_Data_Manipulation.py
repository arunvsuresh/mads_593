import sys
# Pandas required for dataframes
# Numpy required for pandas
# Pyarrow installed to write/read parquet files
# Pip upgrade required due to pandas wheel build issues

# required_libs = ['pandas==1.2.5', 'numpy==1.23.4', 'pyarrow==9.0.0']
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


# <hr style="border:1px solid gray">

# ### Data Manipulation

# #### Zillow

# Let's first start with our Zillow data. For our analysis, we used Zillow's ZHVI (Zillow Home Value Index) dataset, which provides typical home values per county going back all the way to 2000. Median Home Values composed a large part (weight) of our Affordability Index, a metric we created as a means to measure housing affordability on a per-county basis. We'll get into further detail about the indices used in our analysis in a separate notebook. 

# ##### Read files

# Each dataset was downloaded as a CSV to start. Here we also download a file containing full US state names to replace Zillow's abbreviated State names.

# In[3]:


import pandas as pd

def preprocess_zillow_data(zillow_file, state_names_file):
    '''
    Preprocesses Zillow data by transforming, cleaning, and manipulating the dataset.

    zillow_file: path to Zillow CSV file
    state_names_file: path to state names Parquet file
    returns: preprocessed Zillow dataframe
    '''
    
    # Read in data files
    zillow = pd.read_csv(zillow_file)
    state_names = pd.read_parquet(state_names_file)

    # Rename 'RegionName' column to 'County'
    zillow = zillow.rename(columns={'RegionName': 'County'})

    # Limit years to 2016 onwards
    years_to_include = range(2016, 2023)
    cols_to_drop = [col for col in zillow.columns if any(str(year) in col for year in range(2000, 2016))]

    # Pad State FIPS and Municipal FIPS with leading 0's
    zillow['Fips'] = zillow['StateCodeFIPS'].astype('str').str.zfill(2) + zillow['MunicipalCodeFIPS'].astype('str').str.zfill(3)

    # Drop non-informative columns
    cols_to_drop.extend(['SizeRank', 'RegionType', 'StateName', 'RegionID', 'Metro'])
    zillow = zillow.drop(columns=cols_to_drop)

    # Convert Zillow df from wide to long format
    zillow = zillow.melt(id_vars=['County', 'State', 'StateCodeFIPS', 'MunicipalCodeFIPS', 'Fips'], var_name='Year', value_name='Home Value')
    zillow['Year'] = zillow['Year'].apply(lambda x: x[:4])
    zillow.Year = zillow.Year.astype('int')

    # Group Zillow df and calculate median home value
    zillow = zillow.groupby(['County', 'State', 'Year', 'StateCodeFIPS', 'MunicipalCodeFIPS', 'Fips'])['Home Value'].median().to_frame('Median Home Value').reset_index()

    # Use state_names df to replace abbreviated state names with full names
    state_names['state'] = state_names['state'].str.strip()
    state_names['code'] = state_names['code'].str.strip()
    zillow['State'] = zillow['State'].replace(list(state_names['code']), list(state_names['state']))
    zillow['State'] = zillow['State'].str.strip()

    # Apply string manipulations to fix county names
    zillow['County'] = zillow['County'].str.replace("County", "", regex=False)
    zillow['County'] = zillow['County'].mask(zillow['State'] == 'Alaska', zillow['County'].str.replace('Borough', ''))
    zillow['County'] = zillow['County'].mask(zillow['State'] == 'Louisiana', zillow['County'].str.replace('Parish', ''))
    zillow['County'] = zillow['County'].str.strip()
    zillow['County'] = zillow['County'].str.replace("De Kalb", "DeKalb")
    zillow['County'] = zillow['County'].str.replace("Dekalb", "DeKalb")
    zillow['County'] = zillow['County'].str.replace("La Salle", "LaSalle")
    zillow['County'] = zillow['County'].str.replace("Dewitt", "DeWitt")
    zillow['County'] = zillow['County'].str.replace("De Soto", "DeSoto")
    zillow['County'] = zillow['County'].str.replace("O Brien", "O'Brien")

    # Drop rows with NaN values
    zillow = zillow.dropna()

    # Round 'Median Home Value' column to two decimal places
    zillow['Median Home Value'] = zillow['Median Home Value'].apply(lambda x: round(x, 2))

    return zillow

zillow = preprocess_zillow_data('data/Zillow_All_Homes_TimeSeries_Smoothed_Seasonally_Adjusted_By_County.csv', 'data/state_names.parquet')



# In[11]:


zillow['Median Home Value'].describe()


# In[12]:


# write file to parquet to maintain the dtypes and for faster read/write
zillow.to_parquet('data/zillow.parquet')


# #### Equifax

# Now, onto the Equifax data. The Equifax dataset used provides county level shelter and utilities data for the year 2022 and their respective projections for the year 2027. Utilities such as gas and electricity are provided, but for the purposes of our analysis, we will take the sum of all utilities on a county/state/year level. Total utilties form another component of our main Affordability Index metric used to measure housing affordability. 

# ##### Read files

# Here we're reading the Zillow data as well since we're using the Zillow counties as the base counties for our analysis, since Zillow contains the home values.

# In[13]:


import pandas as pd

def preprocess_equifax_data(zillow, equifax_file):
    '''
    Preprocesses Equifax data by transforming, cleaning, and manipulating the dataset.

    zillow: preprocessed Zillow dataframe
    equifax_file: path to Equifax CSV file
    returns: preprocessed Equifax dataframe
    '''
    
    equifax = pd.read_csv(equifax_file)

    # List unique Zillow counties to filter Equifax data
    counties_to_include = list(zillow['County'].unique())

    # Drop unnecessary columns
    equifax = equifax.drop(columns=['Average Mortgage Interest', 'Average Property Tax', 'Average Monthly Housing Payment', 'Average Household Income'])

    # Fix spelling of State names
    equifax['State'].replace("Conneticut", "Connecticut", inplace=True)
    equifax['State'].replace("Deleware", "Delaware", inplace=True)

    # Manipulate string columns and create Total Utility Costs
    equifax['Median Household Income'] = equifax['Median Household Income'].str.replace("$", "").str.replace(",", "").astype("float")
    equifax['Total Households'] = equifax['Total Households'].str.replace("$", "").str.replace(",", "").astype("float")
    equifax['Total Shelter'] = equifax['Total Shelter'].str.replace("$", "").str.replace(",", "").astype("float")
    equifax.iloc[:, 6:] = equifax.iloc[:, 6:].applymap(lambda x: x.replace("$", "").replace(",", "")).astype('float')
    equifax['Total Utility Costs'] = equifax.iloc[:, 6:].sum(axis=1)
    equifax['Total Utility Costs'] = equifax['Total Utility Costs'].apply(lambda x: round(x, 2))
    equifax = equifax.drop(columns=['Total Shelter'])

    # Drop rows with NaN values
    equifax = equifax.dropna(subset=['Median Household Income', 'Total Utility Costs'])

    # Fix county and state strings
    equifax['County'] = equifax['County'].str.strip()
    equifax['State'] = equifax['State'].str.strip()

    # Remove the words 'County', 'Parish', 'Borough' from the end of County field name
    equifax['County'] = equifax['County'].apply(lambda x: x.replace("County", "").replace("Parish", "").replace("Borough", "")).str.strip()

    # Filter Equifax data to only include counties housed in the Zillow dataset
    equifax = equifax[equifax['County'].isin(counties_to_include)]

    return equifax

equifax = preprocess_equifax_data(zillow, 'data/Equifax_Consumer_Expenditure_Shelter_Utilities_Detail_Comparisons.csv')



# In[18]:


# write out equifax data to parquet to maintain dtypes and for faster read/write
equifax_filepath = 'data/equifax.parquet'
equifax.to_parquet(equifax_filepath)


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
    Loop through each CSV year file, read it into a dataframe,
    and concatenate each dataframe into a final df

    year_files: list containing each CSV year file
    returns: final, concatenated dataframe
    '''
    types = defaultdict(str, income='float')
    dfs = []

    for year in year_files:
        year_value = int(year[:4])  # Extract the year from the filename
        
        if year_value >= 2016 and year_value <= 2021:
            df = pd.read_csv('data/' + year, dtype=types, keep_default_na=False)
            df['Year'] = year_value
            dfs.append(df)
        else:
            print("Skipping invalid year:", year)

    # Concatenate all dataframes into one
    final_df = pd.concat(dfs, ignore_index=True)
    
    return final_df



# Now, let's create a function to clean the concatenated dataset by only keeping Median Income, State & County. We'll melt our data from wide to long as each column represented a different county (not helpful when trying to merge by State/County/Year) and clean our County and State strings.

# In[21]:


def clean_acs_data(df):
    '''
    Filters and cleans the concatenated ACS dataframe

    df: final concatenated ACS dataframe
    returns: cleaned ACS dataframe with median household income
    '''
    # Strip leading and trailing whitespaces in 'Label (Grouping)' column
    df['Label (Grouping)'] = df['Label (Grouping)'].str.strip()

    # Select relevant columns based on column names
    income_columns = [col for col in df.columns if '!!Median income (dollars)!!Estimate' in col]
    relevant_columns = income_columns + ['Label (Grouping)', 'Year']

    # Filter the dataframe to only include rows with 'Households' in 'Label (Grouping)'
    df_filtered = df[df['Label (Grouping)'] == 'Households'][relevant_columns]

    # Reshape the filtered dataframe using the 'melt' function
    df_melted = df_filtered.melt(id_vars=['Label (Grouping)', 'Year'], var_name='County', value_name='Median Household Income')

    # Extract state and county information from the 'County' column
    df_melted['County'] = df_melted['County'].str.replace("!!Median income (dollars)!!Estimate", "", regex=False)
    df_melted['State'] = df_melted['County'].str.split(",").apply(lambda x: x[1].strip())
    df_melted['County'] = df_melted['County'].str.split(",").apply(lambda x: x[0].strip()).str.replace("County", "", regex=False).str.strip()

    # Clean 'Median Household Income' column and convert to float
    df_melted['Median Household Income'] = df_melted['Median Household Income'].str.replace(",", "").str.extract("([0-9]+)").astype('float')

    return df_melted



def process_acs_data(year_files, counties_to_include):
    '''
    Process ACS data by appending, cleaning, and filtering the dataset.

    year_files: list containing each CSV year file
    counties_to_include: list of counties to filter
    returns: processed ACS dataframe
    '''

    # Append ACS data
    acs = append_acs_data(year_files)

    # Clean ACS data
    acs_cleaned = clean_acs_data(acs)

    # Fix specific County names
    acs_cleaned['County'] = acs_cleaned['County'].mask(acs_cleaned['State'] == 'Alaska', acs_cleaned['County'].str.replace('Borough', '')).str.strip()
    acs_cleaned['County'] = acs_cleaned['County'].mask(acs_cleaned['State'] == 'Alaska', acs_cleaned['County'].str.replace('Municipality', '')).str.strip()
    acs_cleaned['County'] = acs_cleaned['County'].mask(acs_cleaned['State'] == 'Louisiana', acs_cleaned['County'].str.replace('Parish', '')).str.strip()
    acs_cleaned['County'] = acs_cleaned['County'].str.replace("De Witt", "DeWitt")
    acs_cleaned['County'] = acs_cleaned['County'].str.replace("De Soto", "DeSoto")

    # Filter ACS dataset
    acs_final = acs_cleaned[acs_cleaned['County'].isin(counties_to_include)]

    # Drop the 'Label (Grouping)' field
    acs_final = acs_final.drop(columns=['Label (Grouping)'])

    return acs_final



# Let's inspect our ACS data for any nans or outliers

# In[23]:

acs = process_acs_data(year_files)
print('ACS nans')
print(acs[acs.isna().any(axis=1)].shape[0] / acs.shape[0])
print(acs.describe())


# The nan percentage is very small, so let's drop nan's here. We'll also write our file out to parquet.

# In[24]:


acs = acs.dropna()

acs_filepath = 'data/acs_5_year.parquet'
acs.to_parquet(acs_filepath)

