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
data = np.random.normal(loc=0, scale=1, size=30)

# Create a pandas DataFrame
df = pd.DataFrame(data, columns=['x1'])

# Define the control limits
qda.ControlCharts.IMR(df, 'x1', K = 3, subset_size = 20)
