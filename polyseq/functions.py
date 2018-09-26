import pandas as pd
import numpy as np

from .expression_matrix import ExpressionMatrix

def concat(*exp_mats):
    if isinstance(exp_mats[0], (list, tuple, np.ndarray)):
        exp_mats = exp_mats[0]
    df = pd.concat(exp_mats, keys=range(len(exp_mats)), names=("sample", "cell"))
    return ExpressionMatrix(df)
