"""
    Useful functions
    Creator: Anton Kainulainen
    Date: 2022-11-30
"""

import numpy as np
import pandas as pd

# label = what d label is
# nLst  = list of what d to X_input
def createX_input(df, dLst, label):
    clmn = df.columns
    X_out = []
    for d in dLst:
        X_out.append(df[clmn[d]])
    X_out.append(df[clmn[label]])
    return X_out


def train2tt(df,test_size): #input: df & percentage size
    # example input(data_df,0.5):
    # sampling indices for training
    np.random.seed(1)
    trainI = np.random.choice(df.shape[0], size=round(len(df)*test_size), replace=False)  # returns random index numbers
    trainIndex = df.index.isin(trainI)  # return True/False list, true at index number chosen above

    # the two sets of data
    train = df.iloc[~trainIndex]  # training set
    test = df.iloc[trainIndex]  # test set
    return train, test
