# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import qda
import statsmodels.api as sm

# Import the dataset
data = pd.read_csv('tests\ESE4_ex4.csv')

# fit model ARIMA with constant term
model = qda.Models.ARIMA(data['EXE4'], order=(1,0,0), add_constant=True)

print('method 1')
qda.Summary.ARIMA(model)

print('method 2')
qda.Summary.auto(model)