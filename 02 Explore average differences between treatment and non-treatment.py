#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:28:51 2022

@author: emmettsexton
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import csv
import seaborn as sns

os.chdir("/Users/emmettsexton/Dropbox/Mac/Desktop/Questions/2022.03.14 Mathematica challenge")

# Load appended and merged practice and practice-year level data
df = pd.read_csv(os.getcwd() + "/Temp/01 Practice and practice-year level.csv")

pre_years = [1, 2]
post_years = [3, 4]
df_pre = df[df.year.isin(pre_years)]
df_post = df[df.year.isin(post_years)]

# Weird pre and post V4_avg
# Potential area under both curves overlap/proportion of whole metric for each variable-file to eyeball
#   common support at the variable-file level. We could then plot the density distributions
#   for this metric across files


# Visualize the share of practices that are treated (event fraction).
# Divided by four because there are 4 years of observation per practice in this data and divided by 500 because each
# simulation outcome has 500 fractions.
event_fractions = df.groupby("file_num")["Z"].sum()/4/500
plt.hist(event_fractions, bins = 25)
plt.show()


# Create visualiztions of covariate balance using the weighted and standardized differences between treated and non-treated averages for each variable
# The averages are grouped by treatment status ("Z", 1 is treated and 0 is non-treated) and the simulation number ("file_num")
df_wt_avg = df_pre.groupby(["Z", "file_num"]).apply(lambda x: pd.Series(np.average(x[["Y", "V1_avg", "V2_avg", "V3_avg", "V4_avg", "V5_A_avg", "V5_B_avg", "V5_C_avg"]], weights = x["n.patients"], axis = 0), ["Y", "V1_avg", "V2_avg", "V3_avg", "V4_avg", "V5_A_avg", "V5_B_avg", "V5_C_avg"])).reset_index()

# Reshape average 
df_wt_avg_long = pd.melt(df_wt_avg, id_vars = ['Z','file_num'], value_vars= ["Y", "V1_avg", "V2_avg", "V3_avg", "V4_avg", "V5_A_avg", "V5_B_avg", "V5_C_avg"])
df_wt_avg_wide = df_wt_avg_long.pivot(index = ['file_num', 'variable'], columns = 'Z', values = 'value').reset_index()

# Rename numericaly named columns
df_diffs = df_wt_avg_wide.rename(columns = {1:"treated", 0:"non-treated"})

# Calculate difference between treatment and non-treatment average variable values standardized by the average of the treatment and non-treatment averages
# This is done as an initial way to vizualize the "common support"/covariate balance
df_diffs['std_diff'] = (df_diffs["treated"] - df_diffs["non-treated"])/((df_diffs["treated"] + df_diffs["non-treated"])/2)


# Density Plot exploration
#sns.displot(df_diffs[df_diffs["variable"] != "V4_avg"], x = "std_diff", kind = "kde", hue = "variable")

# Plot difference between treatment/non-treatment averages by variable
fig, axes = plt.subplots(3, 3, figsize=(10,10), dpi=1000)

for i, (ax, variable) in enumerate(zip(axes.flatten(), df_diffs.variable.unique())):
    x = df_diffs.loc[df_diffs.variable==variable, 'std_diff']
    ax.hist(x, alpha=0.5, bins=30, density=True, stacked=True, label=str(variable))
    ax.axvline(0, color = "r")
    
    ax.set_title(variable)   

plt.title('Hists of treatment/non-treatment differences divided by mean between treatment/non-treatment', y = 3.5, loc = "right")

