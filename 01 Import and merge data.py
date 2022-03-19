#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:04:40 2022

@author: emmettsexton
"""
import pandas as pd
import glob
import os
import csv

os.chdir("/Users/emmettsexton/Dropbox/Mac/Desktop/Questions/2022.03.14 Mathematica challenge")

# Load practice (time invariant) level data. 3400 simulations with 500 practices per simulation
# Practice IDs are not consistent across simulations

practice_path = os.getcwd() + "/Source/practice/"
practice_files = glob.glob(practice_path + "/*.csv")
practice_list = []

for filename in practice_files:
    df= pd.read_csv(filename, index_col=None, header=0)
    df["file_num"] = filename[-8:-4]
    practice_list.append(df)

df_practice = pd.concat(practice_list, axis=0, ignore_index=True)

# Load practice-year level data (4 years of data for each practice)
# Practice IDs are not consistent across simulations

practice_year_path = os.getcwd() + "/Source/practice_year"
practice_year_files = glob.glob(practice_year_path + "/*.csv")
practice_year_list = []

for filename in practice_year_files:
    df= pd.read_csv(filename, index_col=None, header=0)
    df["file_num"] = filename[-8:-4]
    practice_year_list.append(df)

df_practice_year = pd.concat(practice_year_list, axis=0, ignore_index=True)

# Merge practice and practice-year level data
df = pd.merge(df_practice, df_practice_year, on = ["id.practice", "file_num"], how = "inner")
del df_practice 
del df_practice_year

df.to_csv(os.getcwd() + "/Temp/01 Practice and practice-year level.csv")
