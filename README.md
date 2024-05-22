# Quality Data Analysis

This repository contains the `qda` module, a Python package that provides functions for creating control charts and returning the results of linear regression and ARIMA models. The output is designed to be user-friendly and similar to the output of Minitab.

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
