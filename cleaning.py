#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 22:31:13 2019

@author: kaustubh
"""
import pandas as pd

br = pd.read_csv('wpb.csv')
br['LymphNodeStatus'].replace(0,1)

bins = [-1,7,14,21,28]
labels = ['NoRisk', 'LowRisk','MedRisk','HighRisk']
br['LymphNodeStatus'] = pd.cut(br['LymphNodeStatus'], bins=bins, labels=labels)

bins = [0,1,2,4,10]
labels = ['NoR','LowR','HighR','VHighR']
br['TumorSize'] = pd.cut(br['TumorSize'], bins=bins, labels=labels)

bins = [10,15,17,20,30]
labels = ['LowR','NormtoLowR','HighR','VHighR']
br['MeanRadius'] = pd.cut(br['MeanRadius'], bins=bins, labels=labels)

bins = [71,97,124,153,183]
labels = ['LowR','NtoLowR','HighR','VHighR']
br['MeanPerimeter'] = pd.cut(br['MeanPerimeter'], bins=bins, labels=labels)

bins = [361,560,970,1151,2251]
labels = ['LowR','NtoLowR','HighR','VHighR']
br['MeanArea'] = pd.cut(br['MeanArea'], bins=bins, labels=labels)

bins = [0.0020,0.0029,0.0060,0.0090,0.040]
labels = ['LowR','NtoLowR','HighR','VHighR']
br['SmoothnessSE'] = pd.cut(br['SmoothnessSE'], bins=bins, labels=labels)

bins = [0.010,0.029,0.05,0.09,0.15]
labels = ['LowR','NtoLowR','HighR','VHighR']
br['ConcavitySE'] = pd.cut(br['ConcavitySE'], bins=bins, labels=labels)

bins = [0.005,0.010,0.015,0.025,0.04]
labels = ['LowR','NtoLowR','HighR','VHighR']
br['ConcavePointsSE'] = pd.cut(br['ConcavePointsSE'], bins=bins, labels=labels)

bins = [0,27,50,70,125]
labels = ['VHighRisk', 'HighRisk','MedRisk','LowRisk']
br['Time'] = pd.cut(br['Time'], bins=bins, labels=labels)



export_csv = br.to_csv(r'Categorical.csv', header = True, index = None)
