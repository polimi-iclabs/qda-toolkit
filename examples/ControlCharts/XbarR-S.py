# Import the libraries
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import qdatoolkit as qda

# Define the data
np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=(30, 5))

# Create a pandas DataFrame
df = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# Define the control limits for the Xbar-R chart
qda.ControlCharts.XbarR(df, K = 3, subset_size = 20)

# Define the control limits for the Xbar-S chart
qda.ControlCharts.XbarS(df, K = 3, subset_size = 20)
