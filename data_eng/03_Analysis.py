#!/usr/bin/env python
# coding: utf-8

# # Analysis Notebook
# by Arun Suresh, Nick Wheatley and James Conner

# <hr style="border:2px solid gray">

# ## Subject of Analysis:
# 
# **Problem Statement**
# While remote work has been a feature of  corporate workplaces for decades, it has not been exceptionally common.  The COVID-19 pandemic dramatically changed the face of remote work in the United States within a few short months, due to prohibitions placed on gathering in public spaces.  
# 
# As companies and the workforce adapted to remote work as a primary mechanism of employment between 2020 and 2022, many well paid professionals were no longer tied to short commutes to the office (Coate 2021).
# 
# This led to an exodus from the high home value city centers to the more reasonably
# 
# Info on analysis purpose priced suburbs exurbs, which has had a side effect of increasing exurb median home values due to surging market demand (Markarian 2021).
# 
# **Motivation**
# The purpose of this report is to analyze the effect on affordability caused by this migration from city center to exurb counties between 2020 and 2022.  We will demonstrate this effect using an Affordability Index, comprised of income, home values, and utility costs per county in the United States.  This index will then be used to create a national analysis, as well as a regional analysis for two large city centers.
# 
# The value of this analysis is in identifying the trend that remote work and the resulting metro center exodus has had on exurb communities and city cores, from the perspective of a potential home buyer.  Using the data, notebooks, indexes and methods of analysis we have comprised in this project, readers can analyze any county of interest for their own needs.
# 
# **Regional Concentration**
# The two regions that have been selected for our analysis are the San Francisco Bay area, and the New York City metropolitan area. These two metropolitan areas are classic examples of high cost city centers with a high concentration of professionals, and exhibit the “donut effect” (Ramani & Bloom 2021).
# <br>
# <hr>
# <br>
# 
# _Coate, Patrick. “Remote Work before, during, and after the Pandemic Quarterly Economics Briefing–Q4 2020.” Remote Work Before, During, and After the Pandemic, National Council on Compensation Insurance, 25 Jan. 2021, https://www.ncci.com/SecureDocuments/QEB/QEB_Q4_2020_RemoteWork.html_
# 
# _Markarian, Kevin. “Council Post: The Effects of Remote Work on Real Estate across the U.S.” Forbes, Forbes Magazine, 23 Apr. 2021, https://www.forbes.com/sites/forbesrealestatecouncil/2021/04/23/the-effects-of-remote-work-on-real-estate-across-the-us/_
# 
# _Ramani, Arjun, and Nicholas Bloom. “The Donut Effect: How Covid-19 Shapes Real Estate.” Stanford Institute for Economic Policy Research, Stanford University, Jan. 2021, https://siepr.stanford.edu/publications/policy-brief/donut-effect-how-covid-19-shapes-real-estate_
# 
# 

# <hr style="border:1px solid gray">

# ## Additional Notebooks
# 
# **[Data Manipulation](01_Data_Manipulation.ipynb)**
# 
# **[Data Merging and Index Creation](02_Data_Merging_and_Index_Creation.ipynb)**

# <hr style="border:1px solid gray">

# ## Notebook Setup

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
# Statsmodels required for plotly trendlines

required_libs = ['pandas==1.2.5', 'numpy==1.23.4', 'plotly==5.10.0', 'pyarrow==9.0.0', 'kaleido==0.2.1', 'statsmodels==0.13.5']
required_libs_str = " ".join(required_libs)

current_libs = get_ipython().getoutput('{sys.executable} -m pip freeze')
                    
if len(set(required_libs) - set(current_libs)) != 0:
    print("Missing or different libraries.")
    print("Installing required libraries/versions.")
    get_ipython().system('{sys.executable} -m pip install --upgrade pip -q')
    get_ipython().system('{sys.executable} -m pip install -q {required_libs_str}')
else:
    print("Libraries and versions match.")


# ### Import Libraries/Modules

# In[ ]:


import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import kaleido
from urllib.request import urlopen


# ### Load Data
# 
# The next step is to load the data, and make any slight modifications to the data that we might need for the analysis, such as changing the format of a column, or adding a convenience column.  Any serious modifications of the data should occur within the previous data manipulation or merging notebooks.

# In[ ]:


# Path to data file
data_path = "data/"

# The list of files for this notebook
file_list = ['merged_dataframe.parquet']

# If the data_path var is set, create a list of files in the data_path
if data_path:
    dir_list = os.listdir(data_path)
else:
    print("Error: The data path variable does not exist!")

# Verify all of the files from the file_list exist inside the directory list
if set(file_list).intersection(set(dir_list)):
    # Import the merged data
    # This dataframe is the result of the data manipulation and index creation stages
    df = pd.read_parquet(os.path.join(data_path,file_list[0]))

    # Add a column which concats County & State for convenience
    # This variable is used for the hover name in the choropleth maps
    df['CountyState'] = df['County'] + ", " + df['State']

    # Multiply the percentages by 100 for easier graphing
    df['median_home_value_2yr_change_perc'] = df['median_home_value_2yr_change_perc']*100
    df['median_income_2yr_change_perc'] = df['median_income_2yr_change_perc']*100
else:
    print("Error:  Not all data from the file list is available.")


# In[ ]:


# Validate the data loaded successfully
df[df.Year == 2022].head(5)


# ### Functions

# #### Helper Functions

# These functions will be used to help create some of the variables needed for the analysis.

# In[ ]:


def create_range(s):
    """
    Creates a min/max tuple from a dataframe series

    s: dataframe series
    returns: (min,max)
    """
    return (np.floor(s.min()), np.ceil(s.max()))

def create_center_point(r,pm=0.2):
    """
    Takes a (min,max) range, and finds the centerpoint.
    Also creates plus/minus variables off the centerpoint,
    which can be used in creating color ranges for the maps.

    r: (min,max) tuple
    pm: Plus Minus

    returns: c, centerpoint
    returns: cp, centerpoint plus the pm float
    returns: cm, centerpoint minus the pm float
    """
    c = np.abs(r[0])/(r[1]-r[0])
    cp = c+pm
    cm = c-pm

    return c,cp,cm


# #### Visualization Functions

# The chart functions below are used to reduce the amount of redundant and repeatative code in the notebook.

# In[ ]:


# These functions are meant to be used in series, and are not independent.
# The correct order of operations for the functions are as follows:

# fig = create_<chart_tye>(dataframe, ... )
# fig = update_lo(fig, ... )
# fig = add_anno(fig, ... )
# fig.show()

def create_choropleth(df, c, lab, ttl, h="", locs='FIPS', c_scale="", r_scale=(0,100), h_name="CountyState", bm=False, fb="locations"):
    """
    Creates the base figure for the choropleth map.

    df:  Pandas Dataframe
    c:   Color variable, a Column of the df
    h:   Hover data as a dictionary
    lab: The Labels variable, a Dictionary of "column name":"renamed"
    ttl: Title variable
    bm:  Show or Hide the "base map"
    fb:  Set the fit boundary for the map. Set to False for national map
    c_scale:  The color scale to be used, if different from default
    r_scale:  The range_color variable, a Tuple of (min,max)
    h_name:   The hover_name variable, a Column of the df

    returns: The map figure
    """
    if c_scale == "":
        c_scale = color_scale
    
    if h == "":
        h = hover_basic
    
    f = px.choropleth(df, 
        geojson=counties, 
        locations='FIPS', 
        color=c,
        color_continuous_scale=c_scale,
        range_color=r_scale,
        scope="usa",
        hover_name='CountyState',
        hover_data=h,
        labels=lab,
        basemap_visible=bm,
        fitbounds=fb,
        title=ttl
    )

    return f


def create_scatter(df, c, x, y, lab, ttl, h="", c_scale="", h_name="CountyState", **kwargs):
    """
    Creates the base figure for the scatterplot.

    df:  Pandas Dataframe
    x:   The x axis variable
    y:   The y axis variable
    c:   Color variable, a Column of the df
    h:   Hover data as a dictionary
    lab: The Labels variable, a Dictionary of "column name":"renamed"
    ttl: Title variable
    h_name:   The hover_name variable, a Column of the df

    returns: The scatterplot figure
    """
    if c_scale == "":
        c_scale = color_scale
    
    if h == "":
        h = hover_basic

    f = px.scatter(df, x=x, y=y, color=c, color_continuous_scale=c_scale, hover_name='CountyState',
        hover_data=h, labels=lab, title=ttl, **kwargs)

    return f


# The update functions below are used to reduce the amount of redundant and repeatative code in the notebook.

# In[ ]:


def update_lo(f, dt=10, ts=" %", tp="", tx=0.5, ty=0.9, y=0.8, x=0.7):
    """
    Updates various items of the base plotly fig, including
    the legend, font, and title centering.

    f:  The figure, from create_<function>()
    dt: The dtick variable, float or int
    tx: Title X location
    ty: Title Y location
    ts: Tick suffix
    x:  The x position for the legend
    y:  The y position for the legend

    returns: The plotly figure
    """

    color_text = '#57677B'

    # Update the Base Map
    f.update_layout(
        # Establish the image margins    
        margin={"r":0,"t":0,"l":0,"b":0, "pad":0},

        # Improve the legend
        coloraxis_colorbar=dict(
        thicknessmode="pixels", thickness=10,
        lenmode="pixels", len=300,
        yanchor="top", y=y,
        xanchor="left", x=x,
        ticks="outside", 
        ticksuffix=ts,
        tickprefix=tp,
        dtick=dt),

        # Set background Transparency
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        geo_bgcolor='rgba(0,0,0,0)',

        # Center the Title
        title_x = tx,
        title_y = ty,

        # Update Font
        font=dict(color=color_text)
    )

    return f

def add_anno(f, t, x=0.25, y=0.0):
    """
    Adds an annotation to the figure.

    f: The figure, from update_lo()
    t: The text of the annotation, as an f-string 
    x: The xshift for the anno
    y: The yshift for the anno 

    returns: The plotly figure
    """
    # Add sourcing annotation
    f.add_annotation(
        text = (t), 
        showarrow=False, x = x, y = y , xref='paper', yref='paper', 
        xanchor='left', yanchor='bottom', xshift=0, yshift=0, 
        font=dict(size=10, color="grey"), align="center")
    
    return f


# ### Create Common Variables, Objects and Dataframes

# **Counties GeoJSON**
# 
# This object contains the polygon boundaries of the counties.  It is passed into the Plotly Express Choropleth maps function to satisfy the requirement for the "geojson" variable.

# In[ ]:


### Download a copy of the geojson counties boundary file
# This file is used for the geojson variable in the choropleths
# It provides the polygon data for the counties in the USA
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# **FIPS Variables**
# 
# Per the [FCC](https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt):  _The Federal Information Processing System (FIPS) codes are numbers which uniquely identify geographic areas.  The number of digits in FIPS codes vary depending on the level of geography.  State-level FIPS codes have two digits, county-level FIPS codes have five digits of which the  first two are the FIPS code of the state to which the county belongs._
# 
# We have created 2 lists of FIPS codes below, one for each of the specific regions in our analysis.

# In[ ]:


### Define FIPS codes
# NYC
# Individually list FIPS codes due to crossing 4 state boundaries
nyc_fips = ["36103","36059","36081","36047","36061","36005","36085",
"36071","36079","36119","36087","36079","36111","36027","36105",
"34031","34021","34029","34003","34039","34017","34025","34023",
"34035","34019","34041","34037","42103","09001","09005","09009",
"34013","42089","42095","42017","34005","34027"]

# SF
sf_counties = ['Alameda','Butte','Calaveras','Colusa',
'Contra Costa','El Dorado','Fresno','Lake','Marin','Mendocino',
'Merced','Monterey','Napa','Sacramento','San Benito',
'San Francisco','San Joaquin','San Mateo','Santa Clara',
'Santa Cruz','Solano','Sonoma','Stanislaus','Sutter','Yolo','Yuba']
sf_fips = list(df.loc[(df.State == 'California') & (df.County.isin(sf_counties)),'FIPS'].unique())


# **Additional Dataframes**
# 
# These dataframes aren't required, but they are used for convenience, reduction in duplicated code, and readability of the notebook.  The first step is to identify the columns we want the focused dataframes to have.

# In[ ]:


# Since we have additional columns in the source data that were created and used in the
# intermediate data processing stages, let's reduce the additional dataframes to just
# the columns that we need in the analysis.
cols = ['County','State','CountyState','Median Household Income','Median Home Value',
        'FIPS','Total Utility Costs','income_index','home_price_index',
        'utility_cost_index','afi','afi_growth','total_cost_index']


# 2022 Dataframes

# In[ ]:


### Create dataframes that limit the data to just 2022 and desired columns
# 3 dataframes created: National, NYC, and SF
# NYC and SF are based on the National 2022 df
nat_2022_df = df[df.Year == 2022][cols]
nyc_2022_df = nat_2022_df[nat_2022_df.FIPS.isin(nyc_fips)]
sf_2022_df = nat_2022_df[nat_2022_df.FIPS.isin(sf_fips)]


# 2020 Dataframes

# In[ ]:


### Create dataframes that limit the data to just 2020 and desired columns
# 3 dataframes created: National, NYC, and SF
# NYC and SF are based on the National 2022 df
nat_2020_df = df[df.Year == 2020][cols]
nyc_2020_df = nat_2020_df[nat_2020_df.FIPS.isin(nyc_fips)]
sf_2020_df = nat_2020_df[nat_2020_df.FIPS.isin(sf_fips)]


# 2 Year Differences Dataframes

# In[ ]:


### First establish the limited columns we want to utilize for the diff
# We will add the County and State information back with a merge after the diff
diff_cols = ['Median Household Income','Median Home Value','FIPS','Total Utility Costs',
            'income_index','home_price_index','utility_cost_index','afi','total_cost_index']

### Create dataframes which contains the differences between 2022-2020
# Create the national dataframe
diff_2yr = pd.DataFrame.round(
df[(df.Year == 2022)][diff_cols].set_index('FIPS').sort_index() - \
df[(df.Year == 2020)][diff_cols].set_index('FIPS').sort_index(), 2).reset_index()

# Merge the County and State columns into the dataframe based on the FIPS code
diff_2yr = diff_2yr.merge(df[['County','State','CountyState','FIPS']], how='left', on='FIPS').drop_duplicates()

# Create the regional diff dataframes, using the national dataframe and their respective FIPS lists
# New York and San Fran Diffs
nyc_diff_2yr = diff_2yr[diff_2yr.FIPS.isin(nyc_fips)]
sf_diff_2yr = diff_2yr[diff_2yr.FIPS.isin(sf_fips)]


# **Min/Max Range Variables**
# 
# These variables help create readable graphs by limiting the color scale to only the range of numbers observed in the dataset's variable of interest.  For example, when viewing a regional map which does not contain values lower `20` and higher than `80`, the range of the variable would be `20-80`, instead of `0-100`.

# In[ ]:


### Define FIPS Min/Max values of interest
# National Ranges
nat_median_home_value_2yr_change_perc = create_range(s=df[(df.Year == 2022)]["median_home_value_2yr_change_perc"])
nat_afi_growth = create_range(s=df[(df.Year == 2022)]["afi_growth"])
nat_afi = create_range(s=df[(df.Year == 2022)]["afi"])
nat_median_home_value = create_range(s=df[(df.Year == 2022)]["Median Home Value"])

# National Diffs
nat_afi_diff = create_range(diff_2yr.afi)
nat_median_home_value_diff = create_range(diff_2yr["Median Home Value"])

###

# NYC Ranges
nyc_median_home_value_2yr_change_perc = create_range(s=df[(df.Year == 2022) & (df.FIPS.isin(nyc_fips))]["median_home_value_2yr_change_perc"])
nyc_afi_growth = create_range(s=df[(df.Year == 2022) & (df.FIPS.isin(nyc_fips))]["afi_growth"])
nyc_afi = create_range(s=df[(df.Year == 2022) & (df.FIPS.isin(nyc_fips))]["afi"])
nyc_median_home_value = create_range(s=df[(df.Year == 2022) & (df.FIPS.isin(nyc_fips))]["Median Home Value"])

# NYC Diff Ranges
nyc_afi_diff = create_range(nyc_diff_2yr.afi)
nyc_median_home_value_diff = create_range(nyc_diff_2yr["Median Home Value"])

###

# SF
sf_median_home_value_2yr_change_perc = create_range(s=df[(df.Year == 2022) & (df.FIPS.isin(sf_fips))]["median_home_value_2yr_change_perc"])
sf_afi_growth = create_range(s=df[(df.Year == 2022) & (df.FIPS.isin(sf_fips))]["afi_growth"])
sf_afi = create_range(s=df[(df.Year == 2022) & (df.FIPS.isin(sf_fips))]["afi"])
sf_median_home_value = create_range(s=df[(df.Year == 2022) & (df.FIPS.isin(sf_fips))]["Median Home Value"])

# SF Diff Ranges
sf_afi_diff = create_range(sf_diff_2yr.afi)
sf_median_home_value_diff = create_range(sf_diff_2yr["Median Home Value"])


# **Centerpoint Ranges**
# 
# Centerpoint ranges take the Min/Max tuple and identifies the center of that range, along with values $\pm$ a given float number.  This is used when wanting to make adjustments to the color scale used in the choropleth maps, by using explicit colors attached to these specific points. 

# In[ ]:


### Create the centerpointranges
# National
cpr_nat_afi_diff = create_center_point(nat_afi_diff)
cpr_nat_mhv_diff = create_center_point(nat_median_home_value_diff)

# New York
cpr_nyc_afi = create_center_point(nyc_afi)
cpr_nyc_afi_diff = create_center_point(nyc_afi_diff)
cpr_nyc_mhv_diff = create_center_point(nyc_median_home_value_diff)

# San Francisco
cpr_sf_afi = create_center_point(sf_afi)
cpr_sf_afi_diff = create_center_point(sf_afi_diff)
cpr_sf_mhv_diff = create_center_point(sf_median_home_value_diff)


# **Color Variables**
# 
# The color variables are created to ensure the consistency of the color palette between the diagrams and the report.

# In[ ]:


color_scale = ["#57677b", "#359cdb", "#f4ab17"]
color_text = "#57677B"


# **Hover Variables**
# 
# These are convience variables to be used to reduce the duplication of code.  These are commonly used groups of column names that we would want to see when hovering over plotly visualizations.

# In[ ]:


# Basic columns
hover_basic= {'FIPS':True,
             'Median Household Income':':,.0f',
             'Median Home Value':':,.0f',
             'afi':':.2f'}


# **Plotly Image Download Config**
# 
# This configuration can be used with the fig.show() function for plotly, changing the scaling, resolution, and format of the downloaded file.

# In[ ]:


px_config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': 'new_image',
    'scale': 4 # Multiply title/legend/axis/canvas sizes by this factor
  }
}


# ---------------------------------------

# ## Analysis

# <hr style="border:1px solid gray">

# ### National Region

# Begin by taking a look at the Affordability Index, since much of our analysis is based on it.

# In[ ]:


# Establish data for scatterplot
sp_data = df.loc[df.Year == 2022,[
    'CountyState','State','County','Median Home Value','Median Household Income','afi']]

# Create scatterplot scaling median income by median home value
fig = px.scatter(sp_data, color='afi',
    x='Median Home Value',
    y='Median Household Income',
    color_continuous_scale=color_scale,
    hover_name="CountyState",
    hover_data=['State','County','afi'],
    labels = {'afi':'Affordability'},
    title='US County Affordability in 2022')

# Improve scatterplot layout
fig.update_layout(
    coloraxis_colorbar=dict(thicknessmode="pixels", thickness=10,
        lenmode="pixels", len=200, yanchor="top", y=1, ticks="outside", dtick=10),
    margin={"r":0,"t":25,"l":0,"b":125},
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    title_x = 0.5,
    font=dict(color=color_text),
    width=400, height = 360)

fig.add_annotation(text = (f"""
Data Sources: Equifax Utilities from Gale Business DemographicsNow,<br>
ACS 5 Yr from Census.gov, Zillow Home Index Value from Zillow.com<br>
Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023"""), 
        showarrow=False, x = -.25, y = -0.5 , xref='paper', yref='paper', 
        xanchor='left', yanchor='bottom', xshift=-0, yshift=0, 
        font=dict(size=10, color="grey"), align="center")

fig.show(config=px_config)


# We can make several interesting observations based on this scatterplot.
# 
# * The index is non-linear.  Though it is Min/Max limited between 0 and 100, the data is not evenly scaled between those numbers.
# * The curve of the data resembles a logarithmic growth curve.  The data is skewed towards the high side of affordability, which is reasonable after consideration.  The amount of rural territory in the National Region compared to urban territory would almost automatically guarentee a skew to affordabilty, when predominately looking at housing and income as the drivers.
# * There are some extreme outliers in the data, such as Nantucket, MA (`afi` value of 0), which has a Median Home Value of \\$2.3M, but a Median Household Income of \\$93k.  In general, outliers on the x-axis (Median Home Value) have lower affordability scores.  When the outliers move towards the upper left corner in the diagram, they have higher affordability scores.  Locations like Los Alamos, NM and Kendall, Il (`afi` value of 100) are outliers on the other side of the spectrum, with very high income and moderately priced housing.
# * The county with the lowest median income (at \\$74k) to have a home value of over \\$1M is Maui, Hawaii.

# Let's take a look at some of the differences between the 2020 and 2022 data. Since we already built a dataframe that contains the differences, we'll utilize that dataframe instead of the basic one.
# 
# We'll start by asking an easy question:  How many counties experienced a loss in Median Home Value in that time range?

# In[ ]:


# Counties with Negative MHV between 2020 and 2022
# Only 6 counties in the entire US decreased in Median Home Values!
diff_2yr[diff_2yr['Median Home Value'] < 0][['State','County','Median Home Value']].dropna()


# Only 6 counties had a loss

# In[ ]:


print("How many counties are in the dataset?  ",len(diff_2yr.FIPS.dropna()))


# In[ ]:


# Counties with more than 250,000 increase in MHV
nat_mhv_incr_250k = diff_2yr[diff_2yr['Median Home Value'] > 250000][['State','County','Median Home Value']].dropna()
print("Number of counties with an increase of more than 250k in Median Home Value: ",len(nat_mhv_incr_250k))


# In[ ]:


# Top 5 Counties that increased in Median Home Value within the National Region
nat_mhv_incr_250k.sort_values(['Median Home Value'], ascending=False).head(5)


# In[ ]:


# Are the counties which had the highest increases in MHV grouped at all?
nat_mhv_incr_250k.groupby(['State'])['County'].count().sort_values(ascending=False)


# ---------------------------------------

# Let's take a look at a choropleth map of the National region.  Based on the scatterplot above, we should see quite a bit yellow/orange (ie: high affordability) across the map, but there's an expectation that several of the western states such as California and Colorado will have a significant amount of blue (ie: low affordability) due to the Median Home Value increases noted in the previous cell.
# 
# There are interesting outliers in that cell above as well, such as Montana, Idaho, and Wyoming, which do not normally have a reputation of high land costs except for vacation/retirement areas.

# In[ ]:


# National Choropleth
# AFI 2022
# Scale modified to reflect non-linear index

cs = [(0, color_scale[0]), 
      (0.35, '#7287a1'), 
      (0.75, color_scale[1]), 
      (0.9, '#ffd118'), 
      (1, color_scale[2])]

fig = create_choropleth(df=nat_2022_df, 
                        c="afi",
                        bm=True,
                        fb=False,
                        ttl="National Affordability Index for 2022",
                        lab={'afi':'Affordability<br>Index'}, 
                        c_scale=cs,
                        r_scale=(0,100))

fig = update_lo(fig, dt=10, ts=" pts", tx=0.3, ty=0.9, x=0.9, y=0.8)
fig = add_anno(fig,y=0.03,x=0.14,t=f"""Data Sources: ACS 5 Yr from Census.gov, Equifax Utilities from Gale Business DemographicsNow,<br>and Zillow Home Index Value from Zillow.com | Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# Since the `afi` metric is non-linear and is skewed towards affordability, we modified the color scale here to be more right skewed, so that the more affordable locations are in a smaller colorband. 
# 
# As expected, many western states are colored blue (ie: less affordable) in this diagram.  There are a few reasons for this, the first and most obvious is that some counties are extremely expensive, such as the 9 counties in California that gained more than \\$250k in Median Home Value in the past 2 years.
# 
# Another thing to note is the extremely large county sizes in the western portion of the United States, as compared to the generally much smaller county sizes in the eastern states. This would have the potential ability to affect median values simply due to the overall land area.  Future analysis should probably consider going with a more fine-grained approach, such as zip code instead of FIPS.
# 
# It is easy to pick out concentrations of unaffordable locations, and they are largely where we expect, gathered around cities and coastlines.  Cities such as San Frandisco, Denver, Dallas, Miami, Boston, New York, Seattle, Nashville, and Phoenix are easily located with shades of blue/green, and most of the southern Florida coastline is also blue/green.

# Since we've looked at the `afi` choropleth on a non-linear scale, let's also take a look at how the `afi` has changed in the past two years.  There are two ways we can look at this:
# 
# * A simple difference of `afi` points from 2022-2020.  The `afi` is calculated independently for each year, so this is a comparison of the affordability of the county, but the difference output is independent of all other counties.  This gives a difference in points.
# 
# * A complex difference is to take the `afi` points from 2022-2020, and then put it back through the non-linear indexing algorithm so the metric is rescaled in relation to all other counties in the United States.  This effectively produces a percentile metric for affordability.
# 
# Let's take a look at the complex method first ...

# In[ ]:


# National Choropleth
# AFI Difference Between 2022-2020
# The results indicate the percentile of change for the county, with respect to all USA counties

fig = create_choropleth(df=nat_2022_df, 
                        c="afi_growth",
                        bm=True,
                        fb=False,
                        ttl="National Affordability Index Change<br>As Percentiles from 2020 to 2022",
                        lab={'afi_growth':'Affordability<br>Index Change<br>Percentiles'}, 
                        c_scale=color_scale,
                        r_scale=(0,100))

fig = update_lo(fig, dt=10, ts="%", tx=0.3, ty=0.9, x=0.9, y=0.8)
fig = add_anno(fig,y=0.03,x=0.14,t=f"""Data Sources: ACS 5 Yr from Census.gov, Equifax Utilities from Gale Business DemographicsNow,<br>and Zillow Home Index Value from Zillow.com | Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# This is interesting because you're effectively looking at change.  Flathead, MT changed the most, so its metric is `0` on this scale.  Locations like Natchitoches, LA experienced minimal change, so its metric is `100` on this map.  Let's look at the numbers for those two locations.

# In[ ]:


# First we'll take a look at the results of differences 2022-2020
diff_2yr[diff_2yr.CountyState.isin(['Natchitoches, Louisiana','Flathead, Montana'])][['CountyState', 'Median Home Value', 'Median Household Income', 'afi']]


# In[ ]:


# Then let's determine the percentage of change those numbers represent
df[(df.CountyState.isin(['Natchitoches, Louisiana','Flathead, Montana'])) & (df.Year == 2022)][['CountyState','median_home_value_2yr_change_perc','median_income_2yr_change_perc']]


# The source of the change for Flathead, MT is rather obvious, it's one of the counties that gained more than \\$250k in Median Home Value between 2020 and 2022.  Not only did it gain \\$321k in MHV, but that number represents an 87.66% change in MHV!  The cost of housing in Flathead, MT nearly doubled in two years, while the Median Income increased by less than \\$5K.
# 
# The numbers for Natchitoches, LA are also very interesting.  It did actually have some change, but ultimately that change leveled out.  It gained \\$23.5k in Median Home Value, but that increase was offset by a substantial increase to the Median Income, coming in at \\$21k.  So to summarize, Natchitoches home values increased by 13%, but the county's median income increased by 68.8%.
# 
# Since the Affordability index is weighted with .7 on median home value, and .25 on median income, this balances out from the index's perspective.
# 
# Next let's take a look at the simplier to understand method of calculating affordability change between 2020 and 2022, using a simple difference between `afi` scores.

# In[ ]:


# National Choropleth
# AFI Difference Between 2022-2020
# The results indicate the points of change for the county

cs = [(0, color_scale[0]), 
      (cpr_nat_afi_diff[2]-.15, '#7287a1'), 
      (cpr_nat_afi_diff[0], color_scale[1]), 
      (cpr_nat_afi_diff[1]-.1, '#ffd118'), 
      (1, color_scale[2])]

fig = create_choropleth(df=diff_2yr, 
                        c="afi",
                        bm=True,
                        fb=False,
                        ttl="National Affordability Index Change<br>As Points from 2020 to 2022",
                        lab={'afi':'Affordability<br>Index Change<br>as Points'}, 
                        c_scale=cs,
                        r_scale=nat_afi_diff)

fig = update_lo(fig, dt=2, ts=" pts", tx=0.5, ty=0.9, x=0.9, y=0.8)
fig = add_anno(fig,x=0.14,y=0.03, t=f"""Data Sources: ACS 5 Yr from Census.gov, Equifax Utilities from Gale Business DemographicsNow,<br>and Zillow Home Index Value from Zillow.com | Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# This view of the data makes it easy to understand where things improved from an affordability perspective.  Scores below `0` indicate affordability decreased, while scores above `0` indicate affordablity increased.
# 
# Note that this doesn't mean these locations are expensive or inexpensive as a whole, only that the affordability of the county increased or decreased in 2022 when compared to its `afi` score from 2020.
# 
# The hotspots in yellow are areas that have improved in affordability.  What's notable is the yellow/orange associated some cities in the US, such as New York and San Francisco.  This is where we will be concentrating our analysis on next.

# ---------------------------------------

# ### New York Region

# In[ ]:


# Add the percentage of change to Income and Home Value to the diff dataframe
pcts_df = df[(df.Year == 2022) & df.FIPS.isin(nyc_fips)][['FIPS','median_income_2yr_change_perc','median_home_value_2yr_change_perc']].rename({'median_income_2yr_change_perc':'MHI_pct','median_home_value_2yr_change_perc':'MHV_pct'}, axis=1)
nyc_diff_2yr = pd.merge(nyc_diff_2yr,pcts_df, on='FIPS', how='left')


# We'll start off by taking a look at some of the basic information from the `nyc_2022_df` and `nyc_diff_2yr` dataframes, looking for any interesting trends, outliers, or notable metrics.

# In[ ]:


nyc_diff_2yr[['State', 'County', 'afi', 'Median Home Value', 'Median Household Income', 'MHV_pct', 'MHI_pct']] \
    .sort_values(['afi'], ascending=False).head(5)


# In[ ]:


### Print out some basic information
# How many counties decreased in affordability between 2020 and 2022?
print("NYC Counties decreased in affordability: {} of {} total".format(nyc_diff_2yr[nyc_diff_2yr.afi < 0]['afi'].count(), len(nyc_diff_2yr)))

# Are there any counties which experienced a loss in the `Median Home Value` between 2020 and 2022?
print("NYC Counties which had a decrease in Median Home Value: {} of {} total".format(nyc_diff_2yr[nyc_diff_2yr['Median Home Value'] < 0]['Median Home Value'].count(), len(nyc_diff_2yr)))

# How many counties decreased in income between 2020 and 2022?
print("NYC Counties had a decreased in income: {} of {} total".format(nyc_diff_2yr[nyc_diff_2yr['Median Household Income'] < 0]['Median Household Income'].count(), len(nyc_diff_2yr)))


# This is quite interesting.  No county in the NYC region experienced a loss of median home value between 2020 and 2022, but 7 out of the 36 counties did experience a loss of median income.  
# 
# The Affordability Index (`afi`) is a composite index whose primary weights are `Median Household Income` (.7) and `Median Home Value` (.25), the expectation is that Median Home Value (MHV) will be the primary driver of the `afi` decrease, but `Median Household Income` (MHI) will certainly be a significant factor as well, especially if it doesn't keep pace with MHV.

# Let's take a look at a scatterplot to see Household Incomes by Home Values to see if anything stands out.  Two versions of the scatterplot are provided, one which uses symbols for the states, and another that breaks the states out into a faceted graph.  Both encode the same information, but viewers may prefer one over the other.

# In[ ]:


fig = create_scatter(df=nyc_2022_df,c='afi',x='Median Household Income', y='Median Home Value', symbol='State',
                     ttl='NYC Home Value by Household Income in 2022', lab={'afi':'Affordability<br>Index'})
fig.update_layout(
    coloraxis_colorbar=dict(dtick=5, ticksuffix=" pts", len=.7, y=.32, x=1.01), 
    title_x=0.5, title_y=0.87)
fig.update_traces(marker=dict(size=8))
fig = add_anno(fig, x=0.02, y=-0.21, t=f"""
Data Sources: ACS 5 Yr from Census.gov, Equifax Utilities from Gale Business DemographicsNow,<br>
and Zillow Home Index Value from Zillow.com | Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# In[ ]:


fig = create_scatter(df=nyc_2022_df,c='afi',x='Median Household Income', y='Median Home Value',
                    facet_col='State', facet_col_wrap=2,
                    ttl='NYC Home Value by Household Income in 2022', lab={'afi':'Affordability<br>Index'})

fig.update_layout(coloraxis_colorbar=dict(dtick=5, ticksuffix=" pts"), title_x=0.5, title_y=0.88)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_traces(textposition='top center', marker=dict(size=12))
fig = add_anno(fig, x=0.02, y=-0.21, t=f"""
Data Sources: ACS 5 Yr from Census.gov, Equifax Utilities from Gale Business DemographicsNow,<br>
and Zillow Home Index Value from Zillow.com | Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# New York county is a significant outlier, with a home value of \$1.3M in 2022!  What's interesting about this is that while New York county has the highest Home Value, it doesn't have the highest Income, which is just to the east in Nassau, NY.  Since the income doesn't track with the home value, the `afi` score for New York is only `48.05`!  Meanwhile, other locations such as Middelsex, NJ, which has a similar Income, has an affordability index score of `86.23`, due to a much lower median home value of \$451k.

# Next let's take a look at some of the highs and lows in these 36 counties to see if anything sticks out.  We'll start by looking at income.

# In[ ]:


# Print out some basic high/low numbers for Median Household Income for 2022
tmhhi22 = nyc_2022_df.sort_values(['Median Household Income'], ascending=False)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()
tmhhi20 = nyc_2020_df.sort_values(['Median Household Income'], ascending=False)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()
lmhhi22 = nyc_2022_df.sort_values(['Median Household Income'], ascending=True)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()
lmhhi20 = nyc_2020_df.sort_values(['Median Household Income'], ascending=True)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()

tmhhidiff_usd = nyc_diff_2yr.sort_values(['Median Household Income'], ascending=False)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()
tmhhidiff_pct = nyc_diff_2yr.sort_values(['MHI_pct'], ascending=False)[['County','State','MHI_pct']].head(1).to_numpy()[0].tolist()
lmhhidiff_usd = nyc_diff_2yr.sort_values(['Median Household Income'], ascending=True)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()
lmhhidiff_pct = nyc_diff_2yr.sort_values(['MHI_pct'], ascending=True)[['County','State','MHI_pct']].head(1).to_numpy()[0].tolist()

print("NYC 2022 Mean HH Inc: $",int(nyc_2022_df['Median Household Income'].agg('mean')))
print("NYC 2022 Mean HH Inc Change from 2020: $", int(nyc_diff_2yr['Median Household Income'].agg('mean')))
print("-----")
print("NYC 2022 Top Median HH Inc: {}, {}: ${}".format(tmhhi22[0],tmhhi22[1],tmhhi22[2]))
print("NYC 2020 Top Median HH Inc: {}, {}: ${}".format(tmhhi20[0],tmhhi20[1],tmhhi20[2]))
print("NYC 2022 Bottom Median HH Inc: {}, {}: ${}".format(lmhhi22[0],lmhhi22[1],lmhhi22[2]))
print("NYC 2020 Bottom Median HH Inc: {}, {}: ${}".format(lmhhi20[0],lmhhi20[1],lmhhi20[2]))
print("-----")
print("NYC Largest Increase HH Inc Change by USD:  {}, {}: ${}".format(tmhhidiff_usd[0],tmhhidiff_usd[1],tmhhidiff_usd[2]))
print("NYC Largest Increase HH Inc Change by %:  {}, {}: {}%".format(tmhhidiff_pct[0],tmhhidiff_pct[1],np.round(tmhhidiff_pct[2])))
print("NYC Largest Decrease HH Inc Change by USD:  {}, {}: ${}".format(lmhhidiff_usd[0],lmhhidiff_usd[1],lmhhidiff_usd[2]))
print("NYC Largest Decrease HH Inc Change by %:  {}, {}: {}%".format(lmhhidiff_pct[0],lmhhidiff_pct[1],np.round(lmhhidiff_pct[2])))


# The mean household income for all counties was just shy of \\$88k, which was a \\$2k increase from 2020, a 2.3% overall change in income.  On average, it doesn't look like income was a primary driver of `afi` change in this location.  But there are a couple of outliers in that statement on both the high side and the low side of the numbers.  On the high side, Rockland, NY, just to the northwest of New York,NY, experienced a 9% improvment to median income.  On the other end of the spectrum, Monroe, PA (75 miles West of NY, NY) experienced a 10% loss of income.  The cause of this change is not immediately obvious, despite checking other data source aggregators, such as https://datacommons.org/place/geoId/42089?utm_medium=explore&mprop=income&popt=Person&cpv=age%2CYears15Onwards&hl=en.  This warrants more investigation at a later time to discern the cause of this.
# 
# 

# Next let's move on to Median Home Vale (MHV).

# In[ ]:


# Print out some basic high/low numbers for Median Home Value for 2022
tmhv22 = nyc_2022_df.sort_values(['Median Home Value'], ascending=False)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()
tmhv20 = nyc_2020_df.sort_values(['Median Home Value'], ascending=False)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()
lmhv22 = nyc_2022_df.sort_values(['Median Home Value'], ascending=True)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()
lmhv20 = nyc_2020_df.sort_values(['Median Home Value'], ascending=True)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()

tmhvdiff_usd = nyc_diff_2yr.sort_values(['Median Home Value'], ascending=False)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()
tmhvdiff_pct = nyc_diff_2yr.sort_values(['MHV_pct'], ascending=False)[['County','State','MHV_pct']].head(1).to_numpy()[0].tolist()
lmhvdiff_usd = nyc_diff_2yr.sort_values(['Median Home Value'], ascending=True)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()
lmhvdiff_pct = nyc_diff_2yr.sort_values(['MHV_pct'], ascending=True)[['County','State','MHV_pct']].head(1).to_numpy()[0].tolist()

print("NYC 2022 Mean Home Val: $",int(nyc_2022_df['Median Home Value'].agg('mean')))
print("NYC 2022 Mean Home Val Change from 2020: $", int(nyc_diff_2yr['Median Home Value'].agg('mean')))
print("-----")
print("NYC 2022 Top Median Home Val: {}, {}: ${}".format(tmhv22[0],tmhv22[1],tmhv22[2]))
print("NYC 2020 Top Median Home Val: {}, {}: ${}".format(tmhv20[0],tmhv20[1],tmhv20[2]))
print("NYC 2022 Bottom Median Home Val: {}, {}: ${}".format(lmhv22[0],lmhv22[1],lmhv22[2]))
print("NYC 2020 Bottom Median Home Val: {}, {}: ${}".format(lmhv20[0],lmhv20[1],lmhv20[2]))
print("-----")
print("NYC Largest Increase Home Val Change by USD:  {}, {}: ${}".format(tmhvdiff_usd[0],tmhvdiff_usd[1],tmhvdiff_usd[2]))
print("NYC Largest Increase Home Val Change by %:  {}, {}: {}%".format(tmhvdiff_pct[0],tmhvdiff_pct[1],np.round(tmhvdiff_pct[2])))
print("NYC Smallest Increase Home Val Change by USD:  {}, {}: ${}".format(lmhvdiff_usd[0],lmhvdiff_usd[1],lmhvdiff_usd[2]))
print("NYC Smallest Increase Home Val Change by %:  {}, {}: {}%".format(lmhvdiff_pct[0],lmhvdiff_pct[1],np.round(lmhvdiff_pct[2])))


# In[ ]:


nyc_diff_2yr[nyc_diff_2yr.County=='New York'][['County','State','Median Home Value']]


# The Median Home Value (MHV) changes are much more dramatic than the changes to income for the NYC area between 2020 and 2022.  
# 
# The mean home value change from 2020 was \\$110k, representing a 27% mean increase to the homes in the region, increasing the overall mean to \\$518k!  Naturally, this mean is somewhat skewed due to the presence of New York county and its \\$1.22M 2020 MHV.  But to look a little deeper at NY county, it only increased by the almost exactly the mean improvment of all the counties in this region, with \\$111.5k added in MHV between 2020 and 2022.  That level of increase is only 9% to the MHV, whereas the other counties experinced a mean increase of 27% to the MHV.
# 
# This is part of the "donut effect", outlined by Arjun Ramani and Nicholas Bloom at https://siepr.stanford.edu/publications/policy-brief/donut-effect-how-covid-19-shapes-real-estate.  Essentially, the city core is decreasing, or not increasing as fast as the exurb counties.  This is driven by the increase to remote work from 6% of the workforce to 35% of the workforce (and 57% of all professionals and managers) (Lund, https://www.mckinsey.com/featured-insights/future-of-work/whats-next-for-remote-work-an-analysis-of-2000-tasks-800-jobs-and-nine-countries).
# 
# 

# Let's take a look at the MHV choropleth graphs, one for 2022, and one for the difference in MHV per county.

# In[ ]:


# Regional Choropleth
# Median Home Value 2022

cs = [(0, color_scale[0]), 
      (0.05, '#7287a1'), 
      (0.2, color_scale[1]), 
      (0.5, '#ffd118'), 
      (1, color_scale[2])]

fig = create_choropleth(nyc_2022_df, 
                        c="Median Home Value",
                        h=hover_basic,
                        bm=False,
                        ttl="NYC Median Home Value in 2022",
                        lab={'Median Home Value':'Median<br>Home Value'}, 
                        c_scale=cs,
                        r_scale=nyc_median_home_value)

fig = update_lo(fig, dt=100000, ts="", tp="$", tx=0.54, ty=0.86)
fig = add_anno(fig, x=0.3, y=0.1, t=f"""
Data Sources: Zillow Home Index Value from Zillow.com<br>
Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# In[ ]:


# Regional Choropleth
# Change in Median Home Value from 2020 to 2022

fig = create_choropleth(nyc_diff_2yr, 
                        c="Median Home Value",
                        h=hover_basic,
                        bm=False,
                        ttl="NYC Change in Median Home Value<br>From 2020 to 2022",
                        lab={'Median Home Value':'Changes to Median<br>Home Value in USD'}, 
                        c_scale=color_scale,
                        r_scale=nyc_median_home_value_diff)

fig = update_lo(fig, dt=15000, ts="", tp="$", tx=0.54, ty=0.87)
fig = add_anno(fig, x=0.3, y=0.1, t=f"""
Data Sources: Zillow Home Index Value from Zillow.com<br>
Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# The lack of significant increase to the MHV of the city core counties is evident in these graphs.  The MHV 2022 graph clearly indicates the city core and the directly connected counties to it are the most expensive, but the "NYC Change in Median Home Value" graph shows those the city core had the lowest amount of MHV growth, and the exurb counties like Suffolk, NY and Monmouth, NJ gained \\$180k and \\$170k, respectively.
# 
# That percentage of increase is very interesting for New York county, so let's look at which counties gainst the most MHV increase in this two-year time span.  The `MHV_pct` variable in the `nyc_diff_2yr` dataframe is already precalculated with that information, so let's sort by that variable descending and get the top 10.

# In[ ]:


nyc_diff_2yr.sort_values(['MHV_pct'], ascending=False).head(10)


# Out of the top 10 counties with MVH increases, none are in the core of the NYC metro area (New York, Queens, Bronx, Kings), instead they're all in the exurb "donut" area.  Of course, the MHV in USD is lower in those locations, so there's headroom for the prices to increase as people from the "hole" of the donut sell expensive homes and take the equity from those sales and drive up costs in the more remote locations.
# 
# Based on this information, if we were to perform a choropleth using the `afi` variable, we should see that the city center should have lower scores, because though they haven't seen the type of growth in the past two years that the exurbs have due to the donut effect, the city core is still more expensive than the exurbs.
# 
# If we then perform a comparison of the differrence between the 2020 and 2022 `afi` scores, that map should be mostly inverted, with the city cores seeing the most improvmement in the past two years in terms of affordability, while the exurbs have been decreasing in affordability.  Let's take a look at those choropleth maps!

# In[ ]:


# Regional Choropleth
# AFI 2022

cs = [(0, color_scale[0]), 
      (.4, '#7287a1'), 
      (.65, color_scale[1]), 
      (.85, '#ffd118'), 
      (1, color_scale[2])]

fig = create_choropleth(nyc_2022_df, 
                        c="afi",
                        h=hover_basic,
                        bm=False,
                        ttl="NYC Affordability Index 2022",
                        lab={'afi':'Affordability<br>Index'}, 
                        c_scale=cs,
                        r_scale=nyc_afi)

fig = update_lo(fig, dt=5, ts=" pts", tx=0.54, ty=0.85)
fig = add_anno(fig, x=0.3, y=0.08, t=f"""
Data Sources: Equifax Utilities from Gale Business DemographicsNow,<br>
ACS 5 Yr from Census.gov, Zillow Home Index Value from Zillow.com<br>
Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# With the darker colors indicating low affordability, and yellow/orange indicating higher affordability, there is a clear ring surrounding the city core of more afforable locations.  Historically, it has always been the case that exurbs and surburbs were cheaper than city centers due to the availability of land and the density of the population.  With the changes to remote work instigated by the COVID-19 pandemic, the increase in remote work opportunities has created the situation of an exodus of highly paid professionals moving out of the city core to the more reasonably priced exurbs.  This means if we look at the affordability from a time perspective, we should see the outer ring of the exurbs decreasing in affordability, while the inner core increases in affordability.

# In[ ]:


# Regional Choropleth
# AFI Difference Between 2022-2020
# The results are the points difference between 2022 - 2020

cs = [(0, color_scale[0]), 
      (cpr_nyc_afi_diff[2], '#7287a1'), 
      (cpr_nyc_afi_diff[0], color_scale[1]), 
      (cpr_nyc_afi_diff[1], '#ffd118'), 
      (1, color_scale[2])]

fig = create_choropleth(nyc_diff_2yr, 
                        c="afi",
                        h=hover_basic,
                        bm=False,
                        ttl="NYC Affordability Index Change<br>As Points from 2020 to 2022",
                        lab={'afi':'Affordability<br>Index Change<br>Points'}, 
                        c_scale=cs,
                        r_scale=nyc_afi_diff)

fig = update_lo(fig, dt=2, ts=" pts", tx=0.54, ty=0.88)
fig = add_anno(fig, x=0.3, y=0.08, t=f"""
Data Sources: Equifax Utilities from Gale Business DemographicsNow,<br>
ACS 5 Yr from Census.gov, Zillow Home Index Value from Zillow.com<br>
Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# This graph is looking at the `afi` score across two dimensions of time, taking the difference of 2020 from 2022.  This graph clearly shows that the affordability of the counties that make up the city core is increasing, while the exurb counties are decreasing in affordability.  This is aforementioned "donut effect" in action.  Does viewing the data without the geospatial context provide any value?

# In[ ]:


cs = [(0, color_scale[0]), 
      (cpr_nyc_afi_diff[2], '#7287a1'), 
      (cpr_nyc_afi_diff[0], color_scale[1]), 
      (cpr_nyc_afi_diff[1], '#ffd118'), 
      (1, color_scale[2])]

fig = create_scatter(df=nyc_diff_2yr,c='afi',x='Median Household Income', y='Median Home Value', c_scale=cs, 
                     symbol='State', trendline='ols', trendline_scope="overall",
                     ttl='NYC Home Value by Household Income<br>2 Year Difference (2020 to 2022)', lab={'afi':'Affordability<br>Index'})
fig.update_layout(
    coloraxis_colorbar=dict(dtick=2, ticksuffix=" pts", len=.7, y=.32, x=1.01), 
    title_x=0.5, title_y=0.87)
fig.update_traces(marker=dict(size=8))
fig = add_anno(fig, x=0.00, y=-0.21, t=f"""
Data Sources: ACS 5 Yr from Census.gov, Equifax Utilities from Gale Business DemographicsNow,<br>
and Zillow Home Index Value from Zillow.com | Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# Without the geospatial encoding, the information is interesting, but not as informative.  Observing the trendline does give us the general relationship between MHV and MHI, but the correlation is weak at best.

# ---------------------------------------

# ### San Francisco Region
# 
# We will now conduct this same analysis in the San Francisco Bay Metropolitan Area to see if this *Donut Effect* replicated elsewhere.
# 
# To permit full analysis, we will first calculate the change from 2020 to 2022:

# In[ ]:


# Add the percentage of change to Income and Home Value to the diff dataframe
pcts_df = df[(df.Year == 2022) & df.FIPS.isin(sf_fips)][['FIPS','median_income_2yr_change_perc','median_home_value_2yr_change_perc']].rename({'median_income_2yr_change_perc':'MHI_pct','median_home_value_2yr_change_perc':'MHV_pct'}, axis=1)
sf_diff_2yr = pd.merge(sf_diff_2yr,pcts_df, on='FIPS', how='left')


# In[ ]:


sf_diff_2yr[['State', 'County', 'afi', 'Median Home Value', 'Median Household Income', 'MHV_pct', 'MHI_pct']] \
    .sort_values(['afi'], ascending=False).head(5)


# Now let's observe how affordability changed from 2020 to 2022:

# In[ ]:


### Print out some basic information
# How many counties decreased in affordability between 2020 and 2022?
print("SF Counties decreased in affordability: {} of {} total".format(sf_diff_2yr[sf_diff_2yr.afi < 0]['afi'].count(), len(sf_diff_2yr)))

# Are there any counties which experienced a loss in the `Median Home Value` between 2020 and 2022?
print("SF Counties which had a decrease in Median Home Value (MHV): {} of {} total".format(sf_diff_2yr[sf_diff_2yr['Median Home Value'] < 0]['Median Home Value'].count(), len(sf_diff_2yr)))

# How many counties decreased in income between 2020 and 2022?
print("SF Counties had a decreased in income (MHI): {} of {} total".format(sf_diff_2yr[sf_diff_2yr['Median Household Income'] < 0]['Median Household Income'].count(), len(sf_diff_2yr)))


# Interesting! Out of the 26 counties in the Bay Area, 20 of them decreased in affordability with only 5 counties dropping in MHI and all countries increasing in MHV. Given the fall in affordability, this indicates that home values increased at a rate disproportionate to income in most counties.
# 
# Let's take a closer look at SF MHI and MHV to be sure:

# In[ ]:


# Print out some basic high/low numbers for Median Household Income for 2022
tmhhi22 = sf_2022_df.sort_values(['Median Household Income'], ascending=False)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()
tmhhi20 = sf_2020_df.sort_values(['Median Household Income'], ascending=False)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()
lmhhi22 = sf_2022_df.sort_values(['Median Household Income'], ascending=True)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()
lmhhi20 = sf_2020_df.sort_values(['Median Household Income'], ascending=True)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()

tmhhidiff_usd = sf_diff_2yr.sort_values(['Median Household Income'], ascending=False)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()
tmhhidiff_pct = sf_diff_2yr.sort_values(['MHI_pct'], ascending=False)[['County','State','MHI_pct']].head(1).to_numpy()[0].tolist()
lmhhidiff_usd = sf_diff_2yr.sort_values(['Median Household Income'], ascending=True)[['County','State','Median Household Income']].head(1).to_numpy()[0].tolist()
lmhhidiff_pct = sf_diff_2yr.sort_values(['MHI_pct'], ascending=True)[['County','State','MHI_pct']].head(1).to_numpy()[0].tolist()

print("SF 2022 Mean HH Inc: $",int(sf_2022_df['Median Household Income'].agg('mean')))
print("SF 2022 Mean HH Inc Change from 2020: $", int(sf_diff_2yr['Median Household Income'].agg('mean')))
print("-----")
print("SF 2022 Top Median HH Inc: {}, {}: ${}".format(tmhhi22[0],tmhhi22[1],tmhhi22[2]))
print("SF 2020 Top Median HH Inc: {}, {}: ${}".format(tmhhi20[0],tmhhi20[1],tmhhi20[2]))
print("SF 2022 Bottom Median HH Inc: {}, {}: ${}".format(lmhhi22[0],lmhhi22[1],lmhhi22[2]))
print("SF 2020 Bottom Median HH Inc: {}, {}: ${}".format(lmhhi20[0],lmhhi20[1],lmhhi20[2]))
print("-----")
print("SF Largest Increase HH Inc Change by USD:  {}, {}: ${}".format(tmhhidiff_usd[0],tmhhidiff_usd[1],tmhhidiff_usd[2]))
print("SF Largest Increase HH Inc Change by %:  {}, {}: {}%".format(tmhhidiff_pct[0],tmhhidiff_pct[1],np.round(tmhhidiff_pct[2])))
print("SF Largest Decrease HH Inc Change by USD:  {}, {}: ${}".format(lmhhidiff_usd[0],lmhhidiff_usd[1],lmhhidiff_usd[2]))
print("SF Largest Decrease HH Inc Change by %:  {}, {}: {}%".format(lmhhidiff_pct[0],lmhhidiff_pct[1],np.round(lmhhidiff_pct[2])))


# In[ ]:


# Print out some basic high/low numbers for Median Home Value for 2022
tmhv22 = sf_2022_df.sort_values(['Median Home Value'], ascending=False)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()
tmhv20 = sf_2020_df.sort_values(['Median Home Value'], ascending=False)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()
lmhv22 = sf_2022_df.sort_values(['Median Home Value'], ascending=True)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()
lmhv20 = sf_2020_df.sort_values(['Median Home Value'], ascending=True)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()

tmhvdiff_usd = sf_diff_2yr.sort_values(['Median Home Value'], ascending=False)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()
tmhvdiff_pct = sf_diff_2yr.sort_values(['MHV_pct'], ascending=False)[['County','State','MHV_pct']].head(1).to_numpy()[0].tolist()
lmhvdiff_usd = sf_diff_2yr.sort_values(['Median Home Value'], ascending=True)[['County','State','Median Home Value']].head(1).to_numpy()[0].tolist()
lmhvdiff_pct = sf_diff_2yr.sort_values(['MHV_pct'], ascending=True)[['County','State','MHV_pct']].head(1).to_numpy()[0].tolist()

print("SF 2022 Mean Home Val: $",int(sf_2022_df['Median Home Value'].agg('mean')))
print("SF 2022 Mean Home Val Change from 2020: $", int(sf_diff_2yr['Median Home Value'].agg('mean')))
print("-----")
print("SF 2022 Top Median Home Val: {}, {}: ${}".format(tmhv22[0],tmhv22[1],tmhv22[2]))
print("SF 2020 Top Median Home Val: {}, {}: ${}".format(tmhv20[0],tmhv20[1],tmhv20[2]))
print("SF 2022 Bottom Median Home Val: {}, {}: ${}".format(lmhv22[0],lmhv22[1],lmhv22[2]))
print("SF 2020 Bottom Median Home Val: {}, {}: ${}".format(lmhv20[0],lmhv20[1],lmhv20[2]))
print("-----")
print("SF Largest Increase Home Val Change by USD:  {}, {}: ${}".format(tmhvdiff_usd[0],tmhvdiff_usd[1],tmhvdiff_usd[2]))
print("SF Largest Increase Home Val Change by %:  {}, {}: {}%".format(tmhvdiff_pct[0],tmhvdiff_pct[1],np.round(tmhvdiff_pct[2])))
print("SF Smallest Increase Home Val Change by USD:  {}, {}: ${}".format(lmhvdiff_usd[0],lmhvdiff_usd[1],lmhvdiff_usd[2]))
print("SF Smallest Increase Home Val Change by %:  {}, {}: {}%".format(lmhvdiff_pct[0],lmhvdiff_pct[1],np.round(lmhvdiff_pct[2])))


# Both incomes and home values grew from 2020-2022, however home values grew at a far greater pace than income. Averaged across all Bay Area counties, home prices grew \\$189k (32%) while incomes only grew \\$3k (4%). Given the greater weight given to home values in the affordability index, this would result in a large shift downwards in affordability.
# 
# This is not true for all counties, however. San Mateo MHI increased 10% while its MHV increased at a slower rate than the region leading to improved affordability performance during that time period. Marin county on the other hand increased home values 34% (\\$410k) while income decreased 4% (\\$4k). Marin's change suggests lower-paid individuals likely seeking cheaper housing, or perhaps non-working individuals with home-equity seeking lower home prices elsewhere.
# 
# To better understand how housing and income affects affordability, let's plot a trendline:

# In[ ]:


fig = create_scatter(df=sf_2022_df,c='afi',x='Median Household Income', y='Median Home Value',
                    facet_col='State', facet_col_wrap=2, trendline='ols', trendline_scope="overall",
                    ttl='SF Home Value by Household Income in 2022', lab={'afi':'Affordability<br>Index'})

fig.update_layout(coloraxis_colorbar=dict(dtick=5, ticksuffix=" pts", len=.95, y=.45, x=1.01), title_x=0.5, title_y=0.88)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_traces(textposition='top center', marker=dict(size=12))
fig = add_anno(fig, x=0.02, y=-0.21, t=f"""
Data Sources: ACS 5 Yr from Census.gov, Equifax Utilities from Gale Business DemographicsNow,<br>
and Zillow Home Index Value from Zillow.com | Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# As shown above, althought the two move quite closely and are positively correlated, affordability seems to be the most impacted by housing prices. Let's display this using Sutter and Stanislaus counties:
# 
# - County: Affordability | Housing | Income
# - Sutter: 78.9 | 443k | \\$62.1k
# - Stanislaus : 77.8 | 464k | \\$64.1k
# 
# Sutter's affordability is 1.4% greater than that of Stanislaus despite its income being 3.1% lower. Housing however is 4.5% cheaper in Sutter, driving the strength in affordability. This logic is present in the composite index weightings.
# 
# Knowing that housing has a greater impact than income, let's filter the greatest growth in MHV and how it impacted affordability:

# In[ ]:


sf_diff_2yr.sort_values(['MHV_pct'], ascending=False).head(10)


# As expected, all counties with large housing value growth fell in affordability. Although in many cases, the decline was tapered by an increase in MHI (as was the case for Merced county).
# 
# Now let's confirm our donut effect assumptions by visualizing it in a choropleth map of the Bay. Let's first look at Affordability in 2022:

# In[ ]:


# Regional Choropleth
# AFI 2022

cs = [(0, color_scale[0]), 
      (.4, '#7287a1'), 
      (.65, color_scale[1]), 
      (.85, '#ffd118'), 
      (1, color_scale[2])]

fig = create_choropleth(sf_2022_df, 
                        c="afi",
                        h=hover_basic,
                        bm=False,
                        ttl="SF Affordability Index 2022",
                        lab={'afi':'Affordability<br>Index'}, 
                        c_scale=cs,
                        r_scale=sf_afi)

fig = update_lo(fig, dt=5, ts=" pts", tx=0.54, ty=0.93)
fig = add_anno(fig,x=0.35,y=0.08,t=f"""
Data Sources: Equifax Utilities from Gale Business DemographicsNow,<br>
ACS 5 Yr from Census.gov, Zillow Home Index Value from Zillow.com<br>
Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# In 2022, lower affordability is concentrated around the San Francisco coast, with inland counties more affordable. Let's see how affordability shifted from 2020 to 2022:

# In[ ]:


# Regional Choropleth
# AFI Difference Between 2022-2020
# The results are the points difference between 2022 - 2020

cs = [(0, color_scale[0]), 
      (cpr_sf_afi_diff[2], '#7287a1'), 
      (cpr_sf_afi_diff[0], color_scale[1]), 
      (cpr_sf_afi_diff[1], '#ffd118'), 
      (1, color_scale[2])]

fig = create_choropleth(sf_diff_2yr, 
                        c="afi",
                        h=hover_basic,
                        bm=False,
                        ttl="SF Bay Affordability Index Change<br>As Points from 2020 to 2022",
                        lab={'afi':'Affordability<br>Index Change<br>Points'}, 
                        c_scale=cs,
                        r_scale=sf_afi_diff)

fig = update_lo(fig, dt=2, ts=" pts", tx=0.54, ty=0.93)
fig = add_anno(fig,x=0.35,y=0.08,t=f"""
Data Sources: Equifax Utilities from Gale Business DemographicsNow,<br>
ACS 5 Yr from Census.gov, Zillow Home Index Value from Zillow.com<br>
Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# As discussed previously, counties such as San Mateo and Santa Clara increased in Affordability as their salary growth exceeded surrounding counties, while their home values grew at a pace slower than surrounding areas. Nearby lower-cost counties such as Santa Cruz and Marin county decreased in affordability the most.
# 
# Let's now see if this is driven by housing as discussed previously:

# In[ ]:


# Regional Choropleth
# Median Home Value 2022

cs = [(0, color_scale[0]), 
      (cpr_sf_mhv_diff[2], '#7287a1'), 
      (cpr_sf_mhv_diff[0], color_scale[1]), 
      (cpr_sf_mhv_diff[1], '#ffd118'), 
      (1, color_scale[2])]

fig = create_choropleth(df[(df.Year == 2022) & (df.FIPS.isin(sf_fips))], 
                        c="Median Home Value",
                        bm=False,
                        ttl="SF Bay Median Home Value in 2022",
                        lab={'Median Home Value':'Median<br>Home Value'}, 
                        c_scale=cs,
                        r_scale=sf_median_home_value)

fig = update_lo(fig, dt=100000, ts="", tp="$", tx=0.54, ty=0.91)
fig = add_anno(fig, x=0.35, y=0.1, t=f"""
Data Sources: Zillow Home Index Value from Zillow.com<br>
Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# As expected, the housing choropleth matches the affordability choropleth closely. Let's explore the change in housing costs:

# In[ ]:


# Regional Choropleth
# Change in Median Home Value from 2020 to 2022

cs = [(0, color_scale[0]), 
      (cpr_sf_mhv_diff[2], '#7287a1'), 
      (cpr_sf_mhv_diff[0], color_scale[1]), 
      (cpr_sf_mhv_diff[1], '#ffd118'), 
      (1, color_scale[2])]

fig = create_choropleth(sf_diff_2yr, 
                        c="Median Home Value",
                        h=hover_basic,
                        bm=False,
                        ttl="SF Bay Change in Median Home Value<br>From 2020 to 2022",
                        lab={'Median Home Value':'Changes to Median<br>Home Value in USD'}, 
                        c_scale=cs,
                        r_scale=sf_median_home_value_diff)

fig = update_lo(fig, dt=50000, ts="", tp="$", tx=0.54, ty=0.91)
fig = add_anno(fig, x=0.35, y=0.1, t=f"""
Data Sources: Zillow Home Index Value from Zillow.com<br>
Data Collected in Jan 2023 | Visualizations via Plotly 5.10.0 in Feb 2023""")
fig.show(config=px_config)


# Home values averaged 32% growth across the Bay from 2020-2022. San Francisco, however, only grew 10% during that time, with Marin county to the North growing 34%. Surprisingly, Marin county incomes also fell 4% during this same time period. This suggests that remote workers are not the only cohorts driving the donut effect. Lower-paid property owners leaving urban centers such as San Francisco also contributed to the donut effect, driving home prices in some cases to match those of the urban centers themselves.
# 
# 

# <hr style="border:1px solid gray">

# ### Summary

# National and regional analysis shows movement towards a decrease in affordability in exurb counties and increases in affordability in city cores, as shown in New York City and San Francisco donut effects. The question that now arises is whether this trend will sustain itself despite [rising interest rates](https://www.bankrate.com/real-estate/how-fed-rate-hike-affects-housing/) accompanied by an increasing number of [available housing inventory](https://fred.stlouisfed.org/series/MSACSR) on the market. With return-to-workplace requirements and changes in macroeconomic policies, future analysis should observe whether the affordability trend will continue, or whether a housing value retraction will take place in suburbs and exurbs.

# 

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=295bb911-c56a-4f18-9d98-f03c8b82d5f2' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
