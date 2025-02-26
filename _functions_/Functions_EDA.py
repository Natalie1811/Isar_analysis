#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)
sns.set(style="white", color_codes=True)

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px  

def get_column_names(data):
    """ This function will be used to extract the column names for numerical and categorical variables
    info from the dataset
    input: dataframe containing all variables
    output: num_vars-> list of numerical columns
            cat_vars -> list of categorical columns"""
        
    num_var = data.select_dtypes(include=['int', 'float']).columns
    print()
    print('Numerical variables are:\n', num_var)
    print('-------------------------------------------------')

    categ_var = data.select_dtypes(include=['category', 'object']).columns
    print('Categorical variables are:\n', categ_var)
    print('-------------------------------------------------') 
    return num_var,categ_var
    
    
def percentage_nullValues(data):
    """
    Function that calculates the percentage of missing values in every column of your dataset
    input: data --> dataframe
    """
    null_perc = round(data.isnull().sum() / data.shape[0],3) * 100.00
    null_perc = pd.DataFrame(null_perc, columns=['Percentage_NaN'])
    null_perc= null_perc.sort_values(by = ['Percentage_NaN'], ascending = False)
    return null_perc


# In[26]:


def select_threshold(data, thr):
    """
    Function that  calculates the percentage of missing values in every column of your dataset
    input: data --> dataframe
    
    """
    null_perc = percentage_nullValues(data)
      
    col_keep = null_perc[null_perc['Percentage_NaN'] < thr]
    col_keep = list(col_keep.index)
    print('Columns to keep:',len(col_keep))
    print('Those columns have a percentage of NaN less than', str(thr), ':')
    print(col_keep)
    data_c= data[col_keep]
    
    return data_c


# In[33]:


def fill_na(data):
    """
    Function to fill NaN with mode (categorical variabls) and mean (numerical variables)
    input: data -> df
    """
    for column in data:
        if data[column].dtype != 'object':
            data[column] = data[column].fillna(data[column].mean())  
        else:
            data[column] = data[column].fillna(data[column].mode()[0]) 
    print('Number of missing values on your dataset are')
    print()
    print(data.isnull().sum())
    return data


# In[2]:

def outliers_box(df,nameOfFeature):
    """
    Function to create a BoxPlot and visualise:
    - All Points in the Variable
    - Suspected Outliers in the variable

    """
    trace0 = go.Box(
        y = df[nameOfFeature],
        name = "All Points",
        jitter = 0.3,
        pointpos = -1.8,
        boxpoints = 'all', #define that we want to plot all points
        marker = dict(
            color = 'rgb(7,40,89)'),
        line = dict(
            color = 'rgb(7,40,89)')
    )

    
    trace1 = go.Box(
        y = df[nameOfFeature],
        name = "Suspected Outliers",
        boxpoints = 'suspectedoutliers', # define the suspected Outliers
        marker = dict(
            color = 'rgba(219, 64, 82, 0.6)',
            #outliercolor = 'rgba(219, 64, 82, 0.6)',
            line = dict(
                outlierwidth = 2)),
        line = dict(
            color = 'rgb(8,81,156)')
    )


    data = [trace0,trace1]

    layout = go.Layout(
        title = "{} Outliers".format(nameOfFeature)
    )

    fig = go.Figure(data=data,layout=layout)
    fig.show()
    #fig.write_html("{}_file.html".format(nameOfFeature))

# In[3]:


def corr_coef(data):
    """
    Function aimed to calculate the corrCoef between each pair of variables
    
    input: data->dataframe        
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    num_vars, categ_var = get_column_names(data)
    data_num = data[num_var]
    data_corr = data_num.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(data_corr,
                xticklabels = data_corr.columns.values,
               yticklabels = data_corr.columns.values,
               annot = True, vmax=1, vmin=-1, center=0, cmap= sns.color_palette("RdBu_r", 7))


# In[4]:

def corr_coef_Threshold(df):
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))

    # Draw the heatmap
    sns.heatmap(df.corr(), annot=True, mask = mask, vmax=1,vmin=-1,
                cmap=sns.color_palette("RdBu_r", 7));


def outlier_treatment(df, colname):
    """
    Function that drops the Outliers based on the IQR upper and lower boundaries 
    input: df --> dataframe
           colname --> str, name of the column
    
    """
    
    # Calculate the percentiles and the IQR
    Q1,Q3 = np.percentile(df[colname], [25,75])
    IQR = Q3 - Q1
    
    # Calculate the upper and lower limit
    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)
    
    # Drop the suspected outliers
    df_clean = df[(df[colname] > lower_limit) & (df[colname] < upper_limit)]
    
    print('Shape of the raw data:', df.shape)
    print('..................')
    print('Shape of the cleaned data:', df_clean.shape)
    return df_clean
       
    
def outliers_loop(df_num):
    """
    jsklfjfl
    
    """
    for item in np.arange(0,len(df_num.columns)):
        if item == 0:
            df_c = outlier_treatment(df_num, df_num.columns[item])
        else:
            df_c = outlier_treatment(df_c, df_num.columns[item]) 
    return df_c   


# pseudo-code for creating an own functions:
#1. parameters: `path1, path2, path3`, name of the variable,
#2. creating the paths: path1 = "xy1.csv", path2 = "xy2.csv" ...
#3. loading the dataframes: df1 = pd.read_csv(path1, sep=';') ...
#4. putting them all together in one dataframe: df_<name of the variable> = pd.concat([df1, df2, df3])
#5. check, if sum_rows of each df is the same as in df_<name of the variable>
#6. print out, that it is complete (or not)
#2. return: 1 dataframe, where all data is in one dataframe


def df_concat(path1, path2, seperator):
    """
    it loads the data that is in multiple csv-files
    and put it together in one file by giving the paths and
    the seperator as strings
    """
    path1 = path1
    path2 = path2

    df_1 = pd.read_csv(path1, sep=seperator)
    df_2 = pd.read_csv(path2, sep=seperator)

    df = pd.concat([df_1, df_2])
    
    sum_rows = df_1.shape[0] + df_2.shape[0]

    if df.shape[0] == sum_rows:
        print("dataframe for is completed. There are", sum_rows, "rows.");
    else:
        print("there's a mistake")
        print("Sum of the rows of each df:", sum_rows)
        print("Rows of the new dataframe:", df.shape[0])
    return df      

def df_concat_excel(list_paths, col_name):
    """
    The function returns a DataFrame by concatenating a list of
    excel-files. 

    Parameters:
    ----------
    list : list_paths
            A list of path-names for excel files to be concatenated.
    String : col_name
            The name of the column that is filled.
        
    Returns:
    -------
    DataFrame
        A single DataFrame containing merged data from multiple Excel files
   
    """

    # Create an empty list to store the created DataFrames in the loop
    list_df = []
    count_rows = 0

    # Loop to create multiple DataFrames to be contenated

    for path in list_paths:
        df = pd.read_excel(path)  # Extract Data from an Excel-file into a DataFrame
        list_df.append(df)      # Store the DataFrames in a list
        count_rows = count_rows + df.shape[0]   # Add rows of each DataFrame for later check

    # Concatenating all created DataFrams into one DataFrame
    df_total = pd.concat(list_df)

    if df_total.shape[0] == count_rows:
        print(f"Dataframe for {col_name} is completed. \nThere are {count_rows} rows.");
    else:
        print(f"There's a mistake.\
              \nSum of the rows of each df: {count_rows}\
              \nRows of the new dataframe: {df.shape[0]}")
    return df_total

# function that marks the outliers of a variable using IQR

def mark_outliers_IQR(df, col_name, outlier_col, multiplier=1.5, verbose=True):

    """
    Marks outliers in a DataFrame column using the IQR method.

    This function calculates the interquartile range (IQR) to determine outliers
    in the given column and flags them in the specified outlier column.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze.
    col_name : str
        The name of the column to check for outliers.
    outlier_col : str
        The name of the column where outlier flags will be stored.
    multiplier : float, optional (default=1.5)
        The multiplier for the IQR to determine outlier limits.
    verbose = bool, optional (default=True)
        Whether to print the details of the operation.

    Returns:
    -------
    None
        The function modifies the DataFrame in place by updating the outlier_column
        with 'suspected' for detected outliers.

    Notes:
    -----
    - The function assumes the column to be numeric.
    - Outliers are defined as values outside the range:
        [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
    """

    # Calculate Q1, Q3 and IQR
    Q1 = df[col_name].quantile(.25) 
    Q3 = df[col_name].quantile(.75)
    IQR = Q3 - Q1
    
    # Determine limits    lower_limit = Q1 - 1.5 * IQR
    lower_limit = Q1 - multiplier * IQR
    upper_limit = Q3 + multiplier * IQR
    
    # Mark outliers in the specified column
    df[outlier_col] = 'normal'  # Default value
    outlier_condition = (df[col_name] < lower_limit) | (df[col_name] > upper_limit)
    df.loc[outlier_condition, outlier_col] = 'suspected'

    # Print details if verbose
    if verbose:
        num_outliers = outlier_condition.sum()
        total_values = df[col_name].count()
        print(f"{col_name}:")
        print(f"Lower limit: {lower_limit}, Upper limit: {upper_limit}")
        print(f"Number of outliers: {num_outliers} out of {total_values}")
        print(df[outlier_col].value_counts())
        print("_________________________")


# function to find out differences in medians with or without outliers:

def diff_medians_outliers(df, col_name, outlier_col):
      """
      Compares the medians of columns with or without outliers.
      Parameters:
      ----------
      df : pd.DataFrame
            The DataFrame containing the data to analyze.
      col_name : str
            The name of the column to investigate.
      outlier_col : str
            The name of the column where outlier flags.
        
      Returns:
      -------
      None
            Prints the medians (with and without outliers) and the percentage difference between them.

      Notes:
      -----
      - This function assumes the target column (`col_name`) contains numeric data.
      - The `outlier_col` should flag normal values with 'normal'. Other values are considered outliers.
      """
      # Calculate the median of the entire column (including outliers)
      median_incl = df[col_name].median()
           
      # Calculate the median of the column excluding outliers
      median_excl = df[df[outlier_col] == 'normal'][col_name].median()

      # calculate the differences of both medians in %
      diff_median_perc = (median_incl - median_excl)/median_incl * 100
     
       # Print the results
      print(f"{col_name}:\nMedian:{median_incl}\
         \nMedian without outliers:{median_excl}\
         \nThe Difference is: {round(diff_median_perc,2)}%\
         \n________________")

def diff_means_outliers(df, col_name, outlier_col):
      """
      Compares the means of columns with or without outliers.
      Parameters:
      ----------
      df : pd.DataFrame
            The DataFrame containing the data to analyze.
      col_name : str
            The name of the column to investigate.
      outlier_col : str
            The name of the column where outlier flags.
        
      Returns:
      -------
      None
            Prints the means (with and without outliers) and the percentage difference between them.

      Notes:
      -----
      - This function assumes the target column (`col_name`) contains numeric data.
      - The `outlier_col` should flag normal values with 'normal'. Other values are considered outliers.
      """
      # Calculate the median of the entire column (including outliers)
      mean_incl = df[col_name].mean()
           
      # Calculate the median of the column excluding outliers
      mean_excl = df[df[outlier_col] == 'normal'][col_name].mean()

      # calculate the differences of both medians in %
      diff_mean_perc = (mean_incl - mean_excl)/mean_incl * 100
     
       # Print the results
      print(f"{col_name}:\nMean:{mean_incl}\
         \nMean without outliers:{mean_excl}\
         \nThe Difference is: {round(diff_mean_perc,2)}%\
         \n________________")

def calculate_neighbours_mean(df, column):
    """
    Calculates the mean of the previous and next valid values for a given column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame (must be sorted by date before calling this function).
        column (str): Name of the column where NaNs should be replaced.
    
    Returns:
        pd.Series: A Series containing the mean of the previous and next values.
    
    Notes:
        - The DataFrame should be sorted chronologically before using this function.
        - This function does not modify the original DataFrame; it only returns calculated values.
    """
   
    # Compute the previous and next neighbors
    prev_value = df[column].shift(1)
    next_value = df[column].shift(-1)
    
    # Calculate the mean of previous and next values:
    neighbors_mean = (prev_value + next_value) / 2

    # Falls der letzte Wert im DF NaN ist → ersetze durch prev_value (da kein next_value existiert)
    neighbors_mean.fillna(prev_value, inplace=True)
    
    # Falls der erste Wert im DF NaN ist → ersetze durch next_value (da kein prev_value existiert)
    neighbors_mean.fillna(next_value, inplace=True)
    
    return neighbors_mean