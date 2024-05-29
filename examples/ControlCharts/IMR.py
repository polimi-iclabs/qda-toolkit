# Import the libraries
import numpy as np
import pandas as pd
import qdatoolkit as qda

# Define the data
np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=30)

# Create a pandas DataFrame
df = pd.DataFrame(data, columns=['x1'])

# Define the control limits
qda.ControlCharts.IMR(df, 'x1', K = 3, subset_size = 20)
