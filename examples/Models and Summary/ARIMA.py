#Import the necessary libraries
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import qdatoolkit as qda

#Import the dataset
data = pd.DataFrame({'Ex5': [-0.14,-0.2,-0.33,-0.46,-0.48,-0.4,-0.28,-0.5,-0.45,-0.43,-0.41,-0.4,-0.36,-0.35,-0.21,-0.12,-0.31,-0.44,-0.33,-0.25,-0.28,-0.19,-0.3,-0.29,-0.29,-0.24,-0.08,-0.16,-0.17,-0.33,-0.11,-0.06,-0.13,-0.21,-0.15,-0.18,-0.12,0.02,0.06,-0.03,-0.08,-0.01,0.03,0.03,0.11,0.03,-0.05,-0.12,-0.09,-0.08,-0.15,-0.05,0.02,0.05,-0.16,-0.17,-0.25,0.02,0.09,0.03,-0.01,0,-0.02,0.02,-0.2,-0.18,-0.05,-0.08,-0.12,0.03,-0.05,-0.22,-0.08,0.06,-0.19,-0.15,-0.22,0.03,-0.05,0.05,0.07,0.09,0.06,0.25,0.03,-0.01,0.08,0.27,0.24,0.16,0.33,0.28,0.1,0.1,0.23,0.33,0.2,0.42]})

# Fit the ARIMA model
model = qda.Models.ARIMA(data, order = (1, 0, 0), add_constant=True)

# Print the summary of the ARIMA model
qda.Summary.auto(model)