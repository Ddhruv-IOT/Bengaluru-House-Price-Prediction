# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 22:54:31 2021

@author: ACER
"""

# pylint: disable=E1101
# pylint: disable=E1137
# pylint: disable=E1136
# pylint: disable=C0303
# pylint: disable=W0612

import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = os.path.dirname(__file__) # r"C:/Users/ACER/Desktop/Workspace/spyder/Machine Learning/home price prediction project -1/datafiles/"

def df_desc(df_):
    """ Desc. of data set"""
    print(df_.head(), "\n")
    print(df_.shape, "\n")
    print(df_.isnull().sum(), "\n")

def is_float(x_float):
    """ float checker... for home area """
    try:
        float(x_float)

    except ValueError:
        tokens = x_float.split("-")

        if len(tokens) == 2:
            x_float = float(tokens[0]) + float(tokens[1])

        else:
            print("Error!--", tokens)
            x_float = None
    
    return x_float

def remove_sqft_outliers(df_):
    """ Function used to remove sqft based outliers """
    df_out = pd.DataFrame()
    for key, subdf in df_.groupby('location'):
        m_e = np.mean(subdf.price_per_sqft)
        s_t = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m_e - s_t)) & (subdf.price_per_sqft<=(m_e + s_t))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

def plot_scatter_chart(df_,location):
    """ Plotting charts for 2vs3bhk prices"""
    bhk2 = df_[(df_.location==location) & (df_.rooms==2)]
    bhk3 = df_[(df_.location==location) & (df_.rooms==3)]
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    

def remove_bhk_outliers(df_):
    """ removing price outliers """
    exclude_indices = np.array([])
    for location, location_df in df_.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('rooms'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('rooms'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices, 
                    bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df_.drop(exclude_indices, axis='index')

# reading dataset
df1 = pd.read_csv(f"{os.path.join(path,'datafiles//bengaluru_house_prices.csv')}")
df_desc(df1)

# desc. as per area type
print(df1.groupby("area_type")["area_type"].agg("count"), "\n")

# dropping things that are not very useful
df1.drop(["area_type", "society", "balcony", "availability"], axis=1, inplace=True)

df_desc(df1)

# filling and dropping empty values in dataset
df1["bath"] = df1["bath"].fillna(math.floor(df1["bath"].median()))
df1.dropna(inplace=True)

df_desc(df1)

# exploring size feature
print(df1["size"].unique(), "\n")

#  room size i.e just keeping number of rooms and removing BHK, etc.
df1["rooms"] = df1["size"].apply(lambda x: int(x.split(" ")[0]))
df1.drop(["size"], axis=1, inplace=True)

df_desc(df1)

# understanding relation between rooms and sqft
print(df1["rooms"].unique(), "\n")
print(df1["total_sqft"].unique(), "\n")

# cleaning sqft and getting useful values from it 
df1["total_sqft"] = df1["total_sqft"].apply(is_float)

print(df1.total_sqft[410], "\n") # test for above func ---

# checking the type of sqft column
print(type(df1.total_sqft[0]), "\n")

# dropping the empty values or not useful ones created by above function is_float
df1.dropna(inplace=True)

df_desc(df1)
print(type(df1.total_sqft[0]), "\n")

# gettig price per sqft
# astype -> convert the type of column, in this case str to flaot
df1["price_per_sqft"] = df1["price"]*100000/df1["total_sqft"].astype(float)
df_desc(df1)

# exploaring number of loactions for 1-hot-encoding
print(len(df1.location.unique()), "\n")

# removing any spaces in location name
df1["location"] = df1["location"].apply(lambda x: x.strip())

# seeing number of houses per location
loca = df1.groupby("location")["location"].agg("count").sort_values(ascending=False)

# finding locations having 10 or less than 10 houses
loca_less_then_10 = loca[loca<=10]
print(loca_less_then_10)

# creating "others" block to store loacations having less humber of houses
# as it will not be playing very important role and also reduce 1-hot-encoded columns
df1.location = df1.location.apply(lambda x: 'other' if x in loca_less_then_10 else x)

# finding price of house as per number of rooms
print(df1[df1["total_sqft"].astype(float)/df1["rooms"].astype(float) < 300].head())

# removing outliers i.e very high or low price variations as per number of rooms 
df1 = df1[~(df1["total_sqft"].astype(float)/df1["rooms"].astype(float) < 300)]
df_desc(df1)

# exploring price per sqft
print(df1.price_per_sqft.describe(), "\n")

# removing outliers in this coulmn
df1 = remove_sqft_outliers(df1)
df_desc(df1)

# visualizing prices between 2BHK, 3BHK flats of same loaction to remove outliers if any.
plot_scatter_chart(df1, "Rajaji Nagar")

# removing outliers in princing b/w 2 and 3 bhk falts of same location
df1 = remove_bhk_outliers(df1)

# visualizing again prices between 2BHK, 3BHK flats of same loaction.
plot_scatter_chart(df1, "Rajaji Nagar")
df_desc(df1)

# exploring relation betwwn rooms and number of bathrooms
print(df1.bath.unique(), "\n")

# finding home having more than 9 bathrooms 
print(df1[df1.bath > 9])

# finding homes having bathrooms more than number of (rooms + 1)
num = df1[df1.bath >= df1.rooms + 1]
print(num, "\n", len(num), "\n")

# removing outliers in above data, i.e having more than (bhk+1) bathroom
df1 = df1[df1.bath <= df1.rooms + 1]
df_desc(df1)

# removing coulmns that are not required
df1.drop(["price_per_sqft"], axis=1, inplace=True)
df_desc(df1)

# using 1-hot-enoding on location names
dummies = pd.get_dummies(df1.location)
print(len(dummies))

# removing 1 dummie column to avoid variable trap
dummies = dummies.drop(["other"], axis=1)

# adding dummie var to original dataset
df1 = pd.concat([df1, dummies], axis=1)

# removing location coulmn from dataset
df1.drop(["location"], axis=1, inplace=True)
df_desc(df1)

# saving processed data to file so it can be used later on.
df1.to_csv(os.path.join(path,'datafiles//processed.csv'), index=False)

print(""" *** Data pre-processing and exploration ends here! *** """)
