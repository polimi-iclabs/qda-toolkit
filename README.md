<p align="center">
  <img src="https://raw.githubusercontent.com/polimi-iclabs/qda-toolkit/main/docs/logo.svg" alt="Logo">
</p>

# qda-toolkit: a Python Package for Statistical Process Control and Quality Data Analysis

This repository contains `qda-toolkit`, a Python package that provides functions for creating control charts and statistical models. The output is designed to be user-friendly and similar to the one provided by other popular commercial software for statistical process control and quality data modeling.

## Features

The QDA module contains several classes, each with its own functionality:

1. **ControlCharts**: This class provides several methods for creating control charts (Shewhart, small shifts and multivariate CC):
    - `IMR`: This method creates an Individual and Moving Range (IMR) control chart.
    - `XbarR`: This method creates an X-bar and R control chart.
    - `XbarS`: This method creates an X-bar and S control chart.
    - `EWMA`: This method creates the exponentially weighted moving average (EWMA) control chart for detecting small shifts. 
    - `CUSUM`: This method creates the cumulative sum (CUSUM) control chart for detecting small shifts.
    - `T2hotelling`: This method creates the Hotelling's T^2 control chart for multivariate data.

    Each of these methods returns a dataframe with the calculated control limits and plots the control chart if `plotit` is set to `True`.

2. **constants**: This class provides static methods to get various statistical constants such as `d2`, `d3`, `c4`, `A2`, `D3`, and `D4`. These constants are used by the `ControlCharts` class to design the control charts.

3. **Models**: This class provides the methods to fit an `ARIMA` of any order `(p,d,q)` (i.e., AR(p), I(d), MA(q)).

4. **StepwiseRegression**: This class provides methods for performing stepwise regression, a method of fitting regression models in which the choice of predictive variables is carried out by an automatic procedure. The class provides methods for fitting the model (`fit`), forward selection (`forward_selection`), backward elimination (`backward_elimination`), and summarizing the model (`SWsummary`).

5. **Summary**: This class provides a method for summarizing the results of a linear regression (`regression`) or ARIMA (`ARIMA`) model in a user-friendly way that mimics the output of other popular software for statistical analysis. The output of the summary functions output the final estimates of parameters, the residual sum of squares, and Ljung-Box Chi-Square Statistics, where applicable. 

## Installation

To install the QDA module, you can install it via `pip`.
```
pip install qda-toolkit
```

## Usage

After installation, you can import the `qdatoolkit` module in your Python script as follows:
```
import qdatoolkit as qda
```
Then, you can create an instance of the QDAModule class and use its methods to create control charts and fit models.
```
# control charts
qda.ControlCharts.IMR(dataframe, column_name)

# summary of a linear regression model
qda.Summary.regression(regression_model_object)
```
## License

This project is licensed under the CC BY-NC-SA 4.0 license. See the LICENSE.md file for details.
