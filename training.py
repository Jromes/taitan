# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:53:26 2020

@author: GO
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WKPATH = "C:/Users/GO/Desktop/1/GIT/taitan"

os.chdir(WKPATH)

train_data_name = "train.csv"
train_data_path = "/".join([WKPATH,train_data_name])

test_data_name = "test.csv"
test_data_path = "/".join([WKPATH,test_data_name])

gs_data_name = "gender_submission.csv"
gs_data_path = "/".join([WKPATH,gs_data_name])
