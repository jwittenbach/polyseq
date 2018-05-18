import numpy as np
from sklearn.linear_model import LinearRegression

from polyseq.expression_matrix import ExpressionMatrix


def regress(data, regressors, n_processes=1):

    regressors = np.array(regressors)
    regressors = data[regressors] if regressors.ndim == 1 else regressors

    model = LinearRegression(n_jobs=n_processes)
    model.fit(regressors, data)

    y_hat = model.predict(regressors)
    residuals = data - y_hat

    zscores = (residuals - residuals.mean())/residuals.std()
    return ExpressionMatrix(zscores, columns=data.columns)
