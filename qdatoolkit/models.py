import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA as arimafromlib
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.graphics.tsaplots as sgt
import matplotlib.pyplot as plt
from scipy import stats
import warnings

def summary(results):
    """Prints a summary of the regression results.

    Parameters
    ----------
    results : RegressionResults object
        The results of a regression model.

    Returns
    -------
    None
    """

    # Set the precision of the output
    np.set_printoptions(precision=4, suppress=True)
    pd.options.display.precision = 4

    # Extract information from the result object
    terms = results.model.exog_names
    coefficients = results.params
    std_errors = results.bse
    t_values = results.tvalues
    p_values = results.pvalues
    #r_squared = results.rsquared
    #adjusted_r_squared = results.rsquared_adj

    # Print the regression equation
    print("REGRESSION EQUATION")
    print("-------------------")
    equation = ("%s = " % results.model.endog_names)
    for i in range(len(coefficients)):
        if results.model.exog_names[i] == 'Intercept':
            equation += "%.3f" % coefficients[i]
        else:
            if coefficients[i] > 0:
                equation += " + %.3f %s" % (coefficients[i], terms[i])
            else:
                equation += " %.3f %s" % (coefficients[i], terms[i])
    print(equation)

    # Print the information in a similar format to Minitab
    print("\nCOEFFICIENTS")
    print("------------")
    # make a dataframe to store the results

    df_coefficients = pd.DataFrame({'Term': terms, 'Coef': coefficients, 'SE Coef': std_errors, 'T-Value': t_values, 'P-Value': p_values})
    df_coefficients.style.format({'Coef': '{:.3f}', 'SE Coef': '{:.3f}', 'T-Value': '{:.3f}', 'P-Value': '{:.3f}'})
    print(df_coefficients.to_string(index=False))

    # Print the R-squared and adjusted R-squared
    print("\nMODEL SUMMARY")
    print("-------------")
    # compute the standard deviation of the distance between the data values and the fitted values
    S = np.std(results.resid, ddof=len(terms))
    # make a dataframe to store the results
    df_model_summary = pd.DataFrame({'S': [S], 'R-sq': [results.rsquared], 'R-sq(adj)': [results.rsquared_adj]})
    print(df_model_summary.to_string(index=False))

    # Print the ANOVA table
    print("\nANALYSIS OF VARIANCE")
    print("---------------------")
    # make a dataframe with the column names and no data
    df_anova = pd.DataFrame(columns=['Source', 'DF', 'Adj SS', 'Adj MS', 'F-Value', 'P-Value'])
    # add the rows of data
    df_anova.loc[0] = ['Regression', results.df_model, results.mse_model * results.df_model, results.mse_model, results.fvalue, results.f_pvalue]
    jj = 1
    for term in terms:
        if term != 'Intercept':
            # perform the f-test for the term
            f_test = results.f_test(term + '= 0')
            df_anova.loc[jj] = [term, f_test.df_num, f_test.fvalue * results.mse_resid * f_test.df_num, f_test.fvalue * results.mse_resid, f_test.fvalue, f_test.pvalue]
            jj += 1

    df_anova.loc[jj] = ['Error', results.df_resid, results.mse_resid * results.df_resid, results.mse_resid, np.nan, np.nan]

    '''
    # Lack-of-fit
    # compute the number of levels in the independent variables 
    n_levels = results.df_resid
    for term in terms:
        if term != 'Intercept':
            n_levels = np.minimum(n_levels, len(data[term].unique())

    if n_levels < results.df_resid:
        dof_lof = n_levels - len(terms)
        dof_pe = results.df_resid - n_levels
        # compute the SSE for the pure error term
        for 


        df_anova.loc[jj + 1] = ['Lack-of-fit', n_levels - len(terms), np.nan, np.nan, np.nan, np.nan]
    '''

    df_anova.loc[jj + 1] = ['Total', results.df_model + results.df_resid, results.mse_total * (results.df_model + results.df_resid), np.nan, np.nan, np.nan]

    print(df_anova.to_string(index=False))

    return


def ARIMAsummary(results):

    """Prints a summary of the ARIMA results.

    Parameters
    ----------
    results : ARIMA object
        The results of an ARIMA.

    Returns
    -------
    None
    """

    # Set the precision of the output
    np.set_printoptions(precision=4, suppress=True)
    pd.options.display.precision = 4

    # Extract information from the result object
    terms = results.param_names
    coefficients = results.params
    std_errors = results.bse
    t_values = results.tvalues
    p_values = results.pvalues
    n_coefficients = len(coefficients) - 1 #because models givers an additional information on sigma^2 in the list of coefficients


    # get the order of the model
    n_model = results.nobs
    ar_order = results.model.order[0]
    ma_order = results.model.order[2]
    diff_order = results.model.order[1]
    order_model = results.model.order
    order_model_flag = sum(order_model) > 0
    max_order=np.max(results.model.order)

    #get seasonal order vector
    so_model = results.model.seasonal_order
    DIFF_seasonal_order = so_model[1]
    seasonal_model_flag = so_model[3] > 0


    #Model's degrees of freedom
    df_model = (results.nobs - diff_order - DIFF_seasonal_order) - (len(results.params) - 1) #degrees of freedom for the model: (n - d - D) - estimated parameters(p, q, P, Q, constant term)

    print("---------------------")
    print("ARIMA MODEL RESULTS")
    print("---------------------")

    if order_model_flag:
        print(f"ARIMA model order: p={ar_order}, d={diff_order}, q={ma_order}")
    if seasonal_model_flag:
        print(f"Seasonal ARIMA model fit with period {so_model[3]} and order: P={so_model[0]}, D={so_model[1]}, Q={so_model[2]}")


    # Print the information in a similar format to Minitab
    print("\nFINAL ESTIMATES OF PARAMETERS")
    print("-------------------------------")
    # make a dataframe to store the results

    df_coefficients = pd.DataFrame({'Term': terms[0:n_coefficients], 'Coef': coefficients[0:n_coefficients], 'SE Coef': std_errors[0:n_coefficients], 'T-Value': t_values[0:n_coefficients], 'P-Value': p_values[0:n_coefficients]})
    df_coefficients.style.format({'Coef': '{:.3f}', 'SE Coef': '{:.3f}', 'T-Value': '{:.3f}', 'P-Value': '{:.3f}'})
    print(df_coefficients.to_string(index=False))


    # Print the ANOVA table
    print("\nRESIDUAL SUM OF SQUARES")
    print("-------------------------")
    # make a dataframe with the column names and no data
    df_rss = pd.DataFrame(columns=['DF', 'SS', 'MS'])
    # add the rows of data
    SSE = np.sum(results.resid[max_order:]**2)

    df_rss.loc[0] = [df_model, SSE, SSE/df_model]
    print(df_rss.to_string(index=False))


    # Print the information in a similar format to Minitab for LBQ test
    print("\nLjung-Box Chi-Square Statistics")
    print("----------------------------------")
    if len(results.resid[max_order:]) > 48:
        lagvalues = np.array([12, 24, 36, 48])
    elif len(results.resid[max_order:]) > 36:
        lagvalues = np.array([12, 24, 36])
    elif len(results.resid[max_order:]) > 24:
        lagvalues = np.array([12, 24])
    elif len(results.resid[max_order:]) > 12:
        lagvalues = np.array([12])
    else:
        lagvalues = int(np.sqrt(len(results.resid[max_order:])))
    LBQ=acorr_ljungbox(results.resid[max_order:], lags=lagvalues, boxpierce=True)

    df_LBtest = pd.DataFrame({'Lag': lagvalues, 'Chi-Square': LBQ.lb_stat, 'P-Value': LBQ.lb_pvalue})
    df_LBtest.style.format({'Lag': '{:.3f}', 'Chi-Square test': '{:.3f}', 'P-Value': '{:.3f}'})
    print(df_LBtest.to_string(index=False))

    return



class Summary:

    @staticmethod
    def auto(results):
        """Prints a summary of the model results.

        Parameters
        ----------
        results : RegressionResults or ARIMAResults object
            The results of a model.
        
        """
        if isinstance(results, sm.regression.linear_model.RegressionResultsWrapper):
            Summary.regression(results)
        elif isinstance(results, statsmodels.tsa.arima.model.ARIMAResultsWrapper):
            Summary.ARIMA(results)
        else:
            print("The type of the results object is not supported.")
        return

    @staticmethod
    def regression(results):
        """Prints a summary of the regression results.

        Parameters
        ----------
        results : RegressionResults object
            The results of a regression model. 

        Returns
        -------
        None
        """

        # Set the precision of the output
        np.set_printoptions(precision=4, suppress=True)
        pd.options.display.precision = 4

        # Extract information from the result object
        terms = results.model.exog_names
        coefficients = results.params
        std_errors = results.bse
        t_values = results.tvalues
        p_values = results.pvalues
        #r_squared = results.rsquared
        #adjusted_r_squared = results.rsquared_adj

        # Print the regression equation
        print("REGRESSION EQUATION")
        print("-------------------")
        equation = ("%s = " % results.model.endog_names)
        for i in range(len(coefficients)):
            if results.model.exog_names[i] == 'Intercept':
                equation += "%.3f" % coefficients[i]
            else:
                if coefficients[i] > 0:
                    equation += " + %.3f %s" % (coefficients[i], terms[i])
                else:
                    equation += " %.3f %s" % (coefficients[i], terms[i])
        print(equation)

        # Print the information in a similar format to Minitab
        print("\nCOEFFICIENTS")
        print("------------")
        # make a dataframe to store the results

        df_coefficients = pd.DataFrame({'Term': terms, 'Coef': coefficients, 'SE Coef': std_errors, 'T-Value': t_values, 'P-Value': p_values})
        df_coefficients.style.format({'Coef': '{:.3f}', 'SE Coef': '{:.3f}', 'T-Value': '{:.3f}', 'P-Value': '{:.3f}'})
        print(df_coefficients.to_string(index=False))

        # Print the R-squared and adjusted R-squared
        print("\nMODEL SUMMARY")
        print("-------------")
        # compute the standard deviation of the distance between the data values and the fitted values
        S = np.std(results.resid, ddof=len(terms))
        # make a dataframe to store the results
        df_model_summary = pd.DataFrame({'S': [S], 'R-sq': [results.rsquared], 'R-sq(adj)': [results.rsquared_adj]})
        print(df_model_summary.to_string(index=False))

        # Print the ANOVA table
        print("\nANALYSIS OF VARIANCE")
        print("---------------------")
        # make a dataframe with the column names and no data
        df_anova = pd.DataFrame(columns=['Source', 'DF', 'Adj SS', 'Adj MS', 'F-Value', 'P-Value'])
        # add the rows of data
        df_anova.loc[0] = ['Regression', results.df_model, results.mse_model * results.df_model, results.mse_model, results.fvalue, results.f_pvalue]
        jj = 1
        for term in terms:
            if term != 'Intercept':
                # perform the f-test for the term
                f_test = results.f_test(term + '= 0')
                df_anova.loc[jj] = [term, f_test.df_num, f_test.fvalue * results.mse_resid * f_test.df_num, f_test.fvalue * results.mse_resid, f_test.fvalue, f_test.pvalue]
                jj += 1

        df_anova.loc[jj] = ['Error', results.df_resid, results.mse_resid * results.df_resid, results.mse_resid, np.nan, np.nan]

        '''
        # Lack-of-fit
        # compute the number of levels in the independent variables 
        n_levels = results.df_resid
        for term in terms:
            if term != 'Intercept':
                n_levels = np.minimum(n_levels, len(data[term].unique())

        if n_levels < results.df_resid:
            dof_lof = n_levels - len(terms)
            dof_pe = results.df_resid - n_levels
            # compute the SSE for the pure error term
            for 


            df_anova.loc[jj + 1] = ['Lack-of-fit', n_levels - len(terms), np.nan, np.nan, np.nan, np.nan]
        '''

        df_anova.loc[jj + 1] = ['Total', results.df_model + results.df_resid, results.mse_total * (results.df_model + results.df_resid), np.nan, np.nan, np.nan]

        print(df_anova.to_string(index=False))

        return

    @staticmethod
    def ARIMA(results):

        """Prints a summary of the ARIMA results.

        Parameters
        ----------
        results : ARIMA object
            The results of an ARIMA.

        Returns
        -------
        None
        """

        # Set the precision of the output
        np.set_printoptions(precision=4, suppress=True)
        pd.options.display.precision = 4

        # Extract information from the result object
        terms = results.param_names
        coefficients = results.params
        std_errors = results.bse
        t_values = results.tvalues
        p_values = results.pvalues
        n_coefficients = len(coefficients) - 1 #because models givers an additional information on sigma^2 in the list of coefficients


        # get the order of the model
        n_model = results.nobs
        ar_order = results.model.order[0]
        ma_order = results.model.order[2]
        diff_order = results.model.order[1]
        order_model = results.model.order
        order_model_flag = sum(order_model) > 0
        max_order=np.max(results.model.order)

        #get seasonal order vector
        so_model = results.model.seasonal_order
        DIFF_seasonal_order = so_model[1]
        seasonal_model_flag = so_model[3] > 0


        #Model's degrees of freedom
        df_model = (results.nobs - diff_order - DIFF_seasonal_order) - (len(results.params) - 1) #degrees of freedom for the model: (n - d - D) - estimated parameters(p, q, P, Q, constant term)

        print("---------------------")
        print("ARIMA MODEL RESULTS")
        print("---------------------")

        if order_model_flag:
            print(f"ARIMA model order: p={ar_order}, d={diff_order}, q={ma_order}")
        if seasonal_model_flag:
            print(f"Seasonal ARIMA model fit with period {so_model[3]} and order: P={so_model[0]}, D={so_model[1]}, Q={so_model[2]}")


        # Print the information in a similar format to Minitab
        print("\nFINAL ESTIMATES OF PARAMETERS")
        print("-------------------------------")
        # make a dataframe to store the results

        df_coefficients = pd.DataFrame({'Term': terms[0:n_coefficients], 'Coef': coefficients[0:n_coefficients], 'SE Coef': std_errors[0:n_coefficients], 'T-Value': t_values[0:n_coefficients], 'P-Value': p_values[0:n_coefficients]})
        df_coefficients.style.format({'Coef': '{:.3f}', 'SE Coef': '{:.3f}', 'T-Value': '{:.3f}', 'P-Value': '{:.3f}'})
        print(df_coefficients.to_string(index=False))


        # Print the ANOVA table
        print("\nRESIDUAL SUM OF SQUARES")
        print("-------------------------")
        # make a dataframe with the column names and no data
        df_rss = pd.DataFrame(columns=['DF', 'SS', 'MS'])
        # add the rows of data
        SSE = np.sum(results.resid[max_order:]**2)

        df_rss.loc[0] = [df_model, SSE, SSE/df_model]
        print(df_rss.to_string(index=False))


        # Print the information in a similar format to Minitab for LBQ test
        print("\nLjung-Box Chi-Square Statistics")
        print("----------------------------------")
        if len(results.resid[max_order:]) > 48:
            lagvalues = np.array([12, 24, 36, 48])
        elif len(results.resid[max_order:]) > 36:
            lagvalues = np.array([12, 24, 36])
        elif len(results.resid[max_order:]) > 24:
            lagvalues = np.array([12, 24])
        elif len(results.resid[max_order:]) > 12:
            lagvalues = np.array([12])
        else:
            lagvalues = int(np.sqrt(len(results.resid[max_order:])))
        LBQ=acorr_ljungbox(results.resid[max_order:], lags=lagvalues, boxpierce=True)

        df_LBtest = pd.DataFrame({'Lag': lagvalues, 'Chi-Square': LBQ.lb_stat, 'P-Value': LBQ.lb_pvalue})
        df_LBtest.style.format({'Lag': '{:.3f}', 'Chi-Square test': '{:.3f}', 'P-Value': '{:.3f}'})
        print(df_LBtest.to_string(index=False))

        return

def ARIMA(x, order, add_constant):
    """Fits an ARIMA model.

    Parameters
    ----------
    x : data object
    
    order : tuple
        The order of the ARIMA model as (p, d, q)

    add_constant : bool
        True if the model should include a constant term, False otherwise

    Returns
    -------
    None
    """
    p=order[0]
    d=order[1]
    q=order[2]

    if add_constant:
        const_coeff='c'
    else:
        const_coeff='n'


    if d!=0:
        x=x.diff(d)

    results = arimafromlib(x, order=(p,0,q), trend=const_coeff).fit()

    # fixing the wrong values in the ARIMA returned object
    results.model.order = (p,d,q)

    # fixing the wrong residuals and fittedvalues in the ARIMA returned object
    results.resid[:np.max(results.model.order)] = np.nan
    results.fittedvalues[:np.max(results.model.order)] = np.nan

    return results

class Models:
    @staticmethod
    def ARIMA(x, order, add_constant):
        """Fits an ARIMA model.

        Parameters
        ----------
        x : data object
        
        order : tuple
            The order of the ARIMA model as (p, d, q)

        add_constant : bool
            True if the model should include a constant term, False otherwise

        Returns
        -------
        None
        """
        p=order[0]
        d=order[1]
        q=order[2]

        if add_constant:
            const_coeff='c'
        else:
            const_coeff='n'


        if d!=0:
            x=x.diff(d)

        results = arimafromlib(x, order=(p,0,q), trend=const_coeff).fit()

        # fixing the wrong values in the ARIMA returned object
        results.model.order = (p,d,q)

        # fixing the wrong residuals and fittedvalues in the ARIMA returned object
        results.resid[:np.max(results.model.order)] = np.nan
        results.fittedvalues[:np.max(results.model.order)] = np.nan

        return results

# create a class called StepwiseRegression that performs stepwise regression when fitting a model
class StepwiseRegression:

    """Performs stepwise regression.

    Parameters
    ----------
    
    y : array-like
        The dependent variable.
    X : array-like
        The independent variables.
    
    add_constant : bool, optional
        Whether to add a constant to the model. The default is True.
    direction : string, optional
        The direction of stepwise regression. The default is 'both'.
    alpha_to_enter : float, optional
        The alpha level to enter a variable in the forward step. The default is 0.15.
    alpha_to_remove : float, optional
        The alpha level to remove a variable in the backward step. The default is 0.15.
    max_iterations : int, optional
        The maximum number of iterations. The default is 100.

    Returns
    -------
    model_fit : RegressionResults object
        The results of a regression model.

    """

    # initialize the class
    def __init__(self, add_constant = True, direction = 'both', alpha_to_enter = 0.15, alpha_to_remove = 0.15, max_iterations = 100):
        self.add_constant = add_constant
        self.direction = direction
        self.alpha_to_enter = alpha_to_enter
        self.alpha_to_remove = alpha_to_remove
        self.max_iterations = max_iterations
        self.break_loop = False
        self.model_fit = None

    # define a function to fit the model
    def fit(self, y, X):
        self.X = X
        self.y = y
        self.variables_to_include = []

        print('Stepwise Regression')
        print('\n######################################')
        k = 1
        print('### Step %d' % k)
        print('-------------------')

        if self.direction == 'backward':
            self.variables_to_include = list(range(self.X.shape[1]))
            X_test = self.X.iloc[:, self.variables_to_include]
            if self.add_constant:
                X_test = sm.add_constant(X_test)
            self.model_fit = sm.OLS(self.y, X_test).fit()

        while not self.break_loop:
            if self.direction == 'forward':
                self.forward_selection()
            elif self.direction == 'backward':
                self.backward_elimination()
            elif self.direction == 'both':
                self.forward_selection()
                if not self.break_loop:
                    self.backward_elimination()
            else:
                raise ValueError('The direction must be either "both", "forward", or "backward".')

            if k == self.max_iterations:
                self.break_loop = True
                print('Maximum number of iterations reached.')

            if not self.break_loop:
                k += 1
                print('\n######################################')
                print('### Step %d' % k)
                print('-------------------')

        return self

    def forward_selection(self):

        print('Forward Selection')

        selected_pvalue = self.alpha_to_enter
        if len(self.variables_to_include) == 0:
            original_variables = []
        else:
            original_variables = self.variables_to_include

        number_of_variables = len(self.variables_to_include)

        if number_of_variables == self.X.shape[1]:
            self.break_loop = True
            print('All predictors have been included in the model. Exiting stepwise.')
            return self

        # fit the model with the selected variables and add one of the remaining variables at a time
        for i in range(self.X.shape[1]):

            if i not in self.variables_to_include:
                # create a new list called testing_variables that includes the original variables and the new variable
                testing_variables = original_variables.copy()
                testing_variables.append(i)

                X_test = self.X.iloc[:, testing_variables]

                if self.add_constant:
                    X_test = sm.add_constant(X_test)

                model_fit = sm.OLS(self.y, X_test).fit()

                # if the p-value of the new variable is less than the alpha_to_enter level, 
                # add the variable to the list of variables to include
                if model_fit.pvalues[-1] < self.alpha_to_enter and model_fit.pvalues[-1] < selected_pvalue:
                    selected_pvalue = model_fit.pvalues[-1]
                    self.variables_to_include = testing_variables
                    self.model_fit = model_fit

        if len(self.variables_to_include) == number_of_variables:
            self.break_loop = True
            print('\nNo predictor added. Exiting stepwise.')
        else:
            # print(self.model_fit.summary())
            self.SWsummary()

        return self


    def backward_elimination(self):

        print('Backward Elimination')

        original_variables = self.variables_to_include

        # sort the pvalues in descending order and remove the variable with pvalue > alpha_to_remove
        if self.add_constant:
            sorted_pvalues = self.model_fit.pvalues[1:].sort_values(ascending = False)
        else:
            sorted_pvalues = self.model_fit.pvalues.sort_values(ascending = False)

        testing_variables = original_variables.copy()

        for i in range(len(sorted_pvalues)):
            if sorted_pvalues[i] > self.alpha_to_remove:
                variable_to_remove = sorted_pvalues.index[i]
                testing_variables.remove(self.X.columns.get_loc(variable_to_remove))
            else:
                break

        if len(testing_variables) == len(original_variables):
            print('\nNo predictor removed.')
            if self.direction == 'backward':
                self.break_loop = True
            return self

        X_test = self.X.iloc[:, testing_variables]

        if self.add_constant:
            X_test = sm.add_constant(X_test)

        self.model_fit = sm.OLS(self.y, X_test).fit()
        self.SWsummary()

        return self

    def SWsummary(self):
        # Extract information from the result object
        results = self.model_fit
        terms = results.model.exog_names
        coefficients = results.params
        p_values = results.pvalues
        #r_squared = results.rsquared
        #adjusted_r_squared = results.rsquared_adj

        # Print the information in a similar format to Minitab
        print("\nCOEFFICIENTS")
        print("------------")
        # make a dataframe to store the results

        df_coefficients = pd.DataFrame({'Term': terms, 'Coef': coefficients, 'P-Value': p_values})
        print(df_coefficients.to_string(index=False))

        # Print the R-squared and adjusted R-squared
        print("\nMODEL SUMMARY")
        print("-------------")
        # compute the standard deviation of the distance between the data values and the fitted values
        S = np.std(results.resid, ddof=len(terms))
        # make a dataframe to store the results
        df_model_summary = pd.DataFrame({'S': [S], 'R-sq': [results.rsquared], 'R-sq(adj)': [results.rsquared_adj]})
        print(df_model_summary.to_string(index=False))

        return


class Assumptions:
    """Test the normality and independence assumptions on data.

    Parameters
    ----------
    data : DataFrame
        The data to test for assumptions.

    Returns
    -------
    None
    """

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            warnings.warn(
                "A numpy array was passed to the Assumptions class and converted to Pandas Series.\n"
                "Note that all other methods in qda-toolkit only accept Pandas Series or Pandas Dataframe.",
                UserWarning
            )
            data = pd.Series(data)
        self.data = data.dropna()

    def normality(self, qqplot=True, test='shapiro-wilk'):
        """Test the normality of the data.

        Parameters
        ----------
        qqplot : bool, optional
            If True, plots the Q-Q plot. Default is True.
        test : str, optional
            Type of test to perform: 'shapiro-wilk' or 'anderson-darling'. Default is 'shapiro-wilk'.

        Returns
        -------
        stat : float
            Test statistic.
        p_value : float
            P-value of the test.
        """

        if qqplot:
            stats.probplot(self.data, dist="norm", plot=plt)
            plt.show()

        if test == 'shapiro-wilk':
            stat, p_value = stats.shapiro(self.data)
        elif test == 'anderson-darling':
            result = stats.anderson(self.data, dist='norm')
            stat = result.statistic
            if stat >= 0.6:
                p_value = np.exp(1.2937 - 5.709 * stat + 0.0186 * (stat ** 2))
            elif stat >= 0.34:
                p_value = np.exp(0.9177 - 4.279 * stat - 1.38 * (stat ** 2))
            elif stat >= 0.2:
                p_value = 1 - np.exp(-8.318 + 42.796 * stat - 59.938 * (stat ** 2))
            else:
                p_value = 1 - np.exp(-13.436 + 101.14 * stat - 223.73 * (stat ** 2))
        else:
            raise ValueError("Invalid test type. Choose 'shapiro-wilk' or 'anderson-darling'.")

        print(f'{test.capitalize()} test statistic = {stat:.3f}')
        print(f'{test.capitalize()} test p-value = {p_value:.3f}')
        return stat, p_value

    def independence(self, plotit=True, ac_test='runs', lag=None, nlags=None):
        """Test the independence of the data.

        Parameters
        ----------
        plots : bool, optional
            If True, plots the ACF and PACF. Default is True.
        runs_test : bool, optional
            If True, performs the runs test. Default is True.
        ac_test : str, optional
            Type of autocorrelation test to perform: 'runs', 'bartlett' or 'lbq'. Default is 'runs'.
        lag : int, optional
            The lag to use for the autocorrelation test. Default is None. 
            Must be specified if ac_test is 'bartlett' or 'lbq'.
        nlags : int, optional
            The number of lags to use for the ACF and PACF plots. Default is None.

        Returns
        -------
        stat : float
            Test statistic for selected ac_test test.
        p_value : float
            P-value for selected ac_test test.

        """

        stat, p_value = None, None

        if nlags is None:
            nlags = min(len(self.data) // 3, 200)
        else:
            # check if the number of lags is less than the length of the data
            if nlags > len(self.data):
                raise ValueError("The number of lags must be less than the length of the data.")
        
        acf_values, stat_lbq, _ = acf(self.data, nlags = nlags, qstat=True, fft = False)

        # check if the lag is specified for the Bartlett or LBQ test
        if ac_test in ['bartlett', 'lbq'] and lag is None:
            raise ValueError("The lag must be specified for the Bartlett or LBQ test.")

        if ac_test == 'runs':
            stat, p_value = runstest_1samp(self.data, correction=False)
            print(f'Runs test statistic = {stat:.3f}')
            print(f'Runs test p-value = {p_value:.3f}\n')

        elif ac_test == 'bartlett':
            rk = acf_values[lag]
            stat = rk
            p_value = 2 * (1 - stats.norm.cdf(abs(stat) * np.sqrt(len(self.data))))
            print(f'Bartlett test statistic = {stat:.3f}')
            print(f'Bartlett test p-value = {p_value:.3f}')
            
        elif ac_test == 'lbq':
            stat = stat_lbq[lag - 1]
            p_value = 1 - stats.chi2.cdf(stat, lag)
            print(f'LBQ test statistic = {stat:.3f}')
            print(f'LBQ test p-value = {p_value:.3f}')

        if plotit:
            fig, ax = plt.subplots(2, 1, figsize=(10, max(5, nlags // 15)))
            sgt.plot_acf(self.data, lags=nlags, zero=False, ax=ax[0])
            ax[0].set_ylim(-1, 1)
            fig.subplots_adjust(hspace=0.5 if nlags <= 50 else 0.2)
            sgt.plot_pacf(self.data, lags=nlags, zero=False, ax=ax[1], method='ywm')
            ax[1].set_ylim(-1, 1)
            plt.show()

        return stat, p_value
