"""
The purpose of this file is to check the control chart modules. All worked, but it needs renewing especially in pandas
TODO: Review constants class in controlcharts
"""

import numpy as np
import pandas as pd
import qdatoolkit as qda

def test_CC_XbarR():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 5))
    data = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
    cc = qda.ControlCharts.XbarR(data)

def test_CC_XbarS():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 5))
    data = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
    cc = qda.ControlCharts.XbarS(data)

def test_CC_IMR():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 1))
    data = pd.DataFrame(data, columns=['A'])
    cc = qda.ControlCharts.IMR(data, 'A')

def test_CC_CUSUM():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 1))
    data = pd.DataFrame(data, columns=['A'])
    cc = qda.ControlCharts.CUSUM(data, 'A', (10, 0.5))

def test_CC_EWMA():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 1))
    data = pd.DataFrame(data, columns=['A'])
    cc = qda.ControlCharts.EWMA(data, 'A', (0.2))

def test_CC_T2hotelling():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(30, 2))
    data = pd.DataFrame(data, columns=['A', 'B'])
    cc = qda.ControlCharts.T2hotelling(data, ['A', 'B'], (10,3), 0.0027)

# changes: added main() function so that the example can run perfectly run if __name__ == '__main__'. instead of running it down one by one
def main():
    test_CC_XbarR()
    test_CC_XbarS()
    test_CC_IMR()
    test_CC_CUSUM()
    test_CC_EWMA()
    test_CC_T2hotelling()

if __name__ == '__main__':
    main()
    # test_CC_XbarR()
    # test_CC_XbarS()
    # test_CC_IMR()
    # test_CC_CUSUM()
    # test_CC_EWMA()
    # test_CC_T2hotelling()