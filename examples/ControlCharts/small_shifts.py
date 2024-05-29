# Import the libraries
import numpy as np
import pandas as pd
import qdatoolkit as qda

# Define the data
np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=30)

# Create a pandas DataFrame
df = pd.DataFrame(data, columns=['x1'])

# Define the control limits for the CUSUM chart
h = 3
k = 0.5
qda.ControlCharts.CUSUM(df, 'x1', (h, k), subset_size = 20)

# Define the control limits for the EWMA chart
lambda_ = 0.2
qda.ControlCharts.EWMA(df, 'x1', lambda_, subset_size = 20)
