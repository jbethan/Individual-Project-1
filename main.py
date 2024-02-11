# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer


from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as skm
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
url = 'https://github.com/esnt/Data/raw/main/ISLR/Default.csv'
df = pd.read_csv(url)

# %%
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler())
])

Xtrain = pipe.fit_transform(Xtrain)
Xtest = pipe.transform(Xtest)

#pipe.fit(x_train, y_train)
#yhat = pipe.predict(x_test)
# %%