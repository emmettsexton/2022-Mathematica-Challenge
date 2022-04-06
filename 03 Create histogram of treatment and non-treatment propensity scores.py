#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 13:52:55 2022

@author: emmettsexton
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from sklearn.linear_model import LogisticRegression

import glob
import csv
import seaborn as sns

os.chdir("/Users/emmettsexton/Dropbox/Mac/Desktop/Questions/2022.03.14 Mathematica challenge")

# Load appended and merged practice and practice-year level data
df = pd.read_csv(os.getcwd() + "/Temp/01 Practice and practice-year level.csv")

# Collapse each file to one observation per practice to fit logit model
pre_years = [1, 2]
df_pre = df[df.year.isin(pre_years)]

# Take patient weighted average of time variant variables for pre treatment years. These will be used in the logit model
df_logit = df_pre.groupby(["id.practice", "file_num"]).apply(lambda x: pd.Series(np.average(x[["n.patients", "Z","Y", "V1_avg", "V2_avg", "V3_avg", "V4_avg", "V5_A_avg", "V5_B_avg", "V5_C_avg"]], weights = x["n.patients"], axis = 0), ["n.patients", "Z", "Y", "V1_avg", "V2_avg", "V3_avg", "V4_avg", "V5_A_avg", "V5_B_avg", "V5_C_avg"])).reset_index()
del df

df_logit.to_csv(os.getcwd() + "/Temp/02 Pre year averages for logit.csv")

# RHS vars
feature_cols = ["n.patients", "Y", "V1_avg", "V2_avg", "V3_avg", "V4_avg", "V5_A_avg", "V5_B_avg", "V5_C_avg"]

# Select one file to test method
df = df_logit[df_logit["file_num"] == 9]

# Train model with treatment "Z" on LHS
train_x = df[feature_cols]
train_y = df["Z"]
model = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 1)
model.fit(train_x, train_y)

# Pull predicted probabilities of treatment for each practice
scores = pd.DataFrame(model.predict_proba(train_x))
scores = scores.rename(columns = {1:"propensity_for_treatment", 0:"propensity_for_non_treatment"})
scores.reset_index(inplace=True)
scores["index"] = scores["index"] + 1
scores = scores.rename(columns = {"index":"id.practice"})
scores = scores[["id.practice", "propensity_for_treatment"]]

# Merge back on to original data frame to identify which practices were actually treated
df_score = df.merge(scores, on = "id.practice")

# Seperate treat/non-treated for histogram
x_treat = df_score.loc[df_score.Z == 1, "propensity_for_treatment"]
x_non_treated = df_score.loc[df_score.Z == 0, "propensity_for_treatment"]

# Create density plot lines by treatment status for predicted treatment probability 
kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2, 'clip':(0,1)})

plt.figure(figsize=(10,7), dpi= 80)
sns.distplot(x_treat, color="dodgerblue", **kwargs)
ax = sns.distplot(x_non_treated, color="orange", **kwargs)

# Calculate integrals of both curves and create summary statistic of the treatment and non-treatment density curve overlap
area_treated = np.trapz(ax.lines[0].get_ydata(), ax.lines[0].get_xdata())
area_non_treated = np.trapz(ax.lines[1].get_ydata(), ax.lines[1].get_xdata())
ymin = np.minimum(ax.lines[0].get_ydata(), ax.lines[1].get_ydata())
area_overlap = pd.Series(np.trapz(ymin, ax.lines[0].get_xdata())/(area_treated + area_non_treated))

# Initiate series to store overlap values
overlap_series = pd.Series()

# Loop through each file and generate one overlap summary stat for each
for file in df_logit.file_num.unique():
    
    df = df_logit[df_logit["file_num"] == file]
    
    train_x = df[feature_cols]
    train_y = df["Z"]
    model = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 1)
    model.fit(train_x, train_y)
    
    scores = pd.DataFrame(model.predict_proba(train_x))
    scores = scores.rename(columns = {1:"propensity_for_treatment", 0:"propensity_for_non_treatment"})
    scores.reset_index(inplace=True)
    scores["index"] = scores["index"] + 1
    scores = scores.rename(columns = {"index":"id.practice"})
    scores = scores[["id.practice", "propensity_for_treatment"]]
    
    df_score = df.merge(scores, on = "id.practice")
    
    x_treat = df_score.loc[df_score.Z == 1, "propensity_for_treatment"]
    x_non_treated = df_score.loc[df_score.Z == 0, "propensity_for_treatment"]
    
    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2, 'clip':(0,1)})
    
    plt.figure(figsize=(10,7), dpi= 80)
    sns.distplot(x_treat, color="dodgerblue", **kwargs)
    ax = sns.distplot(x_non_treated, color="orange", **kwargs)
    area_treated = np.trapz(ax.lines[0].get_ydata(), ax.lines[0].get_xdata())
    area_non_treated = np.trapz(ax.lines[1].get_ydata(), ax.lines[1].get_xdata())
    ymin = np.minimum(ax.lines[0].get_ydata(), ax.lines[1].get_ydata())
    area_overlap = pd.Series(np.trapz(ymin, ax.lines[0].get_xdata())/(area_treated + area_non_treated))
    
    overlap_series = overlap_series.append(area_overlap, ignore_index = True)

