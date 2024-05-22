# Quality Data Analysis

This repository contains `qda`, a Python package that provides functions for creating control charts and statistical models. The output is designed to be user-friendly and similar to the one provided by other popular commercial software for statistical process control and quality data modeling.

## Features

The QDA module contains several classes, each with its own functionality:

1. **ControlCharts**: This class provides several methods for creating control charts. Here are some of the key methods:
    - `IMR(original_df, col_name, K = 3, subset_size = None, run_rules = False, plotit = True)`: This method creates an Individual and Moving Range (IMR) control chart. It takes in a dataframe, a column name, a constant K, an optional subset size, a boolean to determine if run rules should be applied, and a boolean to determine if the chart should be plotted.
    - `XbarR(original_df, K = 3, subset_size = None, plotit = True)`: This method creates an X-bar and R control chart. It takes in a dataframe, a constant K, an optional subset size, and a boolean to determine if the chart should be plotted.
    - `XbarS(original_df, K = 3, sigma = None, subset_size = None, plotit = True)`: This method creates an X-bar and S control chart. It takes in a dataframe, a constant K, an optional sigma value, an optional subset size, and a boolean to determine if the chart should be plotted.
    - `EWMA`
    - `CUSUM` 

    Each of these methods returns a dataframe with the calculated control limits and plots the control chart if `plotit` is set to `True`.

2. **constants**: This class provides static methods to get various statistical constants such as `d2`, `d3`, `c4`, `A2`, `D3`, and `D4`. These constants are used by the `ControlCharts` class to design the control charts.

3. **StepwiseRegression**: This class provides methods for performing stepwise regression, a method of fitting regression models in which the choice of predictive variables is carried out by an automatic procedure. The class provides methods for fitting the model (`fit`), forward selection (`forward_selection`), backward elimination (`backward_elimination`), and summarizing the model (`SWsummary`).

4. **Summary**: This class provides a method for summarizing the results of a linear regression (`regression`) or ARIMA (`ARIMA`) model in a user-friendly way that mimics the output of other popular software for statistical analysis. The output of the summary functions output the final estimates of parameters, the residual sum of squares, and Ljung-Box Chi-Square Statistics, where applicable. 


## Features

The module is made by two main classes: 

1. **ControlCharts**: it contains the functions to create the most popular control charts for statistical process control. It includes Shewart's control charts for samples (Xbar-R, Xbar-S) and for individuals (I, I-MR), and two popular control charts for small shifts (CUSUM, EWMA). 

2. **Summary**: it contains the functions to output the results of `statsmodels` linear regression and ARIMA models in a user-friendly way and with a layout similar to Minitab. 

## Installation

To install the QDA module, you can install it via `pip`.
```
pip install qda
```

You can also clone this repository to your local machine:
```bash
git clone https://github.com/username/qda-module.git
```

Then, navigate to the cloned directory and install the required dependencies.
```
cd qda-module
pip install -r requirements.txt
```

## Usage

After installation, you can import the QDA module in your Python script as follows:
```
import qda
```
Then, you can create an instance of the QDAModule class and use its methods to create control charts, perform linear regression, and return the results of ARIMA models.
```
qda.ControlCharts.IMR(data)
qda.summary(regression_model)
```
## License

This project is licensed under the MIT License. See the LICENSE.md file for details.
