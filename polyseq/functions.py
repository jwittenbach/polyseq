import pandas as pd

def concat(exp_mats):
    return pd.concat(exp_mats, keys=range(len(exp_mats)), names=("sample", "cell"))
