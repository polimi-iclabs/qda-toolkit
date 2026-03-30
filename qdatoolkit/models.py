import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA as arimafromlib
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.graphics.tsaplots as sgt
import matplotlib.pyplot as plt
from scipy import stats
import warnings


class Summary:
    """Formatted summary printer for statsmodels regression and ARIMA results.

    Produces Minitab-style console output including coefficient tables,
    model fit statistics, ANOVA tables, and diagnostic tests. All methods
    are static; the class is used purely as a namespace.
    """

    @staticmethod
    def fmt_float(x):
        """Format a numeric value with precision based on its magnitude.

        Chooses the number of decimal places dynamically so that small
        values retain meaningful digits while large values stay compact.

        Parameters
        ----------
        x : float, str, or NaN
            The value to format. Empty strings and NaN are returned as "".  

        Returns
        -------
        str
            Formatted string: 2 dp for |x| >= 10, 3 dp for |x| >= 1,
            4 dp for |x| >= 0.01, and scientific notation otherwise.
        """
        if x == "" or not pd.notna(x):
            return ""
        try:
            ax = abs(float(x))
        except (ValueError, TypeError):
            return str(x)
        if ax >= 100:
            return f"{float(x):.2f}"
        elif ax >= 10:
            return f"{float(x):.2f}"
        elif ax >= 1:
            return f"{float(x):.3f}"
        elif ax >= 0.01:
            return f"{float(x):.4f}"
        else:
            return f"{float(x):.3e}"

    @staticmethod
    def fmt_p(x):
        """Format a p-value to exactly 3 decimal places.

        Parameters
        ----------
        x : float, str, or NaN
            The p-value to format. Empty strings and NaN are returned as "".

        Returns
        -------
        str
            The p-value formatted as ``"X.XXX"``, or ``""`` for missing values.
        """
        if x == "" or not pd.notna(x):
            return ""
        else:
            return f"{float(x):.3f}"

    @staticmethod
    def fmt_vif(x):
        """Format a Variance Inflation Factor (VIF) value.

        Similar to :meth:`fmt_float` but uses 2 decimal places for values
        >= 1 (instead of 3) since VIF values don't need sub-unit precision.

        Parameters
        ----------
        x : float, str, or NaN
            The VIF value to format. Empty strings and NaN are returned
            as "" (used for the intercept term, which has no VIF).

        Returns
        -------
        str
            Formatted VIF string: 2 dp for |x| >= 1, 3 dp for |x| >= 0.01,
            and scientific notation otherwise.
        """
        if x == "" or not pd.notna(x):
            return ""
        try:
            ax = abs(float(x))
        except (ValueError, TypeError):
            return str(x)
        if ax >= 100:
            return f"{float(x):.2f}"
        elif ax >= 10:
            return f"{float(x):.2f}"
        elif ax >= 1:
            return f"{float(x):.2f}"
        elif ax >= 0.01:
            return f"{float(x):.3f}"
        else:
            return f"{float(x):.3e}"

    @staticmethod
    def fmt_pct(x):
        """Format a proportion (0-1) as a percentage string.

        Multiplies the value by 100 and appends a ``%`` sign.

        Parameters
        ----------
        x : float, str, or NaN
            A proportion between 0 and 1. Empty strings and NaN are
            returned as "".

        Returns
        -------
        str
            Percentage string with 3 decimal places, e.g. ``"85.234%"``.
        """
        if x == "" or not pd.notna(x):
            return ""
        try:
            return f"{float(x) * 100:.3f}%"
        except (ValueError, TypeError):
            return str(x)

    def fmt_int(x):
        """Format a numeric value as an integer string (no decimal places).

        Parameters
        ----------
        x : float, str, or NaN
            The value to format. Empty strings and NaN are returned as-is.

        Returns
        -------
        str
            The value cast to int and converted to string, e.g. ``"5"``.
        """
        return f"{int(x)}" if pd.notna(x) and x != "" else x

    @staticmethod
    def auto(results):
        """Detect the model type and print the appropriate Minitab-style summary.

        Checks whether ``results`` is an OLS regression or ARIMA results
        object and delegates to :meth:`regression` or :meth:`ARIMA`
        accordingly. Prints an error message if the type is not recognized.

        Parameters
        ----------
        results : RegressionResultsWrapper or ARIMAResultsWrapper
            A fitted model results object from statsmodels.
        """
        if isinstance(results, sm.regression.linear_model.RegressionResultsWrapper):
            Summary.regression(results)
        elif isinstance(results, statsmodels.tsa.arima.model.ARIMAResultsWrapper):
            Summary.ARIMA(results)
        else:
            print("The type of the results object is not supported.")

    @staticmethod
    def regression(results):
        """Print a Minitab-style summary for an OLS regression model.

        FIrst, the datatype is rechecked. In case users went straigh to Summary.regression(results) instead of auto

        Outputs four sections to stdout:
        1. **Regression Equation** -- the fitted equation as a readable string.
        2. **Coefficients** -- a table of term names, coefficients, standard
           errors, t-values, p-values, and Variance Inflation Factors (VIF).
        3. **Model Summary** -- standard error of the regression (S),
           R-squared, and adjusted R-squared.
        4. **Analysis of Variance** -- ANOVA table with regression, per-term,
           error, and total rows showing DF, SS, MS, F-value, and p-value.

        .. warning::
            This method sets ``np.set_printoptions`` and
            ``pd.options.display.precision`` as a side effect.

        Parameters
        ----------
        results : RegressionResultsWrapper
            A fitted OLS model from ``statsmodels.api.OLS(...).fit()``.
        """

        # the first thingk to do is to reverify the instance type
        checker = isinstance(results, sm.regression.linear_model.RegressionResultsWrapper)
        if not checker:
            print("The type of the results object is not supported")
            return None

        # Set the precision of the output
        np.set_printoptions(precision=4, suppress=True)
        pd.options.display.precision = 4

        # Extract information from the result object
        terms = results.model.exog_names
        coefficients = results.params
        std_errors = results.bse
        t_values = results.tvalues
        p_values = results.pvalues
        
        if len(terms) > 1:
            vifs = [variance_inflation_factor(results.model.exog, i) for i in range(len(terms))]

            if terms[0]=='const':
                vifs[0]=""

        else:
            if terms == 'const':
                vifs = [""]
            else:
                vifs = [1]

        # Print the regression equation
        print("REGRESSION EQUATION")
        print("-------------------")
        equation = ("%s = " % results.model.endog_names)
        for i in range(len(coefficients)):
            if results.model.exog_names[i] == 'Intercept':
                equation += "%.3f" % coefficients[i]
            else:
                if coefficients.iloc[i] > 0:
                    equation += " + %.3f %s" % (coefficients.iloc[i], terms[i])
                else:
                    equation += " %.3f %s" % (coefficients.iloc[i], terms[i])
        print(equation)

        print("\nCOEFFICIENTS")
        print("------------")
        df_coefficients = pd.DataFrame({'Term': terms, 'Coef': coefficients, 'SE Coef': std_errors, 'T-Value': t_values, 'P-Value': p_values, 'VIF': vifs})

        print(df_coefficients.to_string(index=False, formatters={
            'Coef': Summary.fmt_float,
            'SE Coef': Summary.fmt_float,
            'T-Value': Summary.fmt_float,
            'P-Value': Summary.fmt_p,
            'VIF': Summary.fmt_vif
        }))

        print("\nMODEL SUMMARY")
        print("-------------")
        S = np.std(results.resid, ddof=len(terms))
        df_model_summary = pd.DataFrame({'S': [S], 'R-sq': [results.rsquared], 'R-sq(adj)': [results.rsquared_adj]})
        print(df_model_summary.to_string(index=False, formatters={
            'S': Summary.fmt_float,
            'R-sq': Summary.fmt_pct,
            'R-sq(adj)': Summary.fmt_pct
        }))

        print("\nANALYSIS OF VARIANCE")
        print("---------------------")
        df_anova = pd.DataFrame(columns=['Source', 'DF', 'Adj SS', 'Adj MS', 'F-Value', 'P-Value'])
        df_anova.loc[0] = ['Regression', results.df_model, results.mse_model * results.df_model, results.mse_model, results.fvalue, results.f_pvalue]
        jj = 1
        for term in terms:
            if term != 'Intercept':
                f_test = results.f_test(term + '= 0')
                df_anova.loc[jj] = [term, f_test.df_num, f_test.fvalue * results.mse_resid * f_test.df_num, f_test.fvalue * results.mse_resid, f_test.fvalue, f_test.pvalue]
                jj += 1

        df_anova.loc[jj] = ['Error', results.df_resid, results.mse_resid * results.df_resid, results.mse_resid, "", ""]
        df_anova.loc[jj + 1] = ['Total', results.df_model + results.df_resid, results.mse_total * (results.df_model + results.df_resid), "", "", ""]
        
        # Ensure P-Value column is numeric, replacing empty strings with NaN
        df_anova['P-Value'] = pd.to_numeric(df_anova['P-Value'], errors='coerce')
        
        # Replace NaN values with empty strings in the P-Value column
        df_anova['P-Value'] = df_anova['P-Value'].fillna("")

        # Ensure all numeric columns are properly formatted
        df_anova['DF'] = df_anova['DF'].apply(Summary.fmt_int)
        df_anova['Adj SS'] = df_anova['Adj SS'].apply(Summary.fmt_float)
        df_anova['Adj MS'] = df_anova['Adj MS'].apply(Summary.fmt_float)
        df_anova['F-Value'] = df_anova['F-Value'].apply(Summary.fmt_float)
        df_anova['P-Value'] = df_anova['P-Value'].apply(Summary.fmt_p)

        print(df_anova.to_string(index=False))

    @staticmethod
    def ARIMA(results):
        """Print a Minitab-style summary for a fitted ARIMA / SARIMA model.

        Outputs four sections to stdout:
        1. **Model Order** -- the (p, d, q) order and, if applicable, the
           seasonal (P, D, Q, s) order.
        2. **Final Estimates of Parameters** -- coefficient table with term
           names, estimates, standard errors, t-values, and p-values. For
           pure AR models without differencing, the constant is adjusted to
           match the Minitab convention: ``constant * (1 - sum(AR coeffs))``.
        3. **Residual Sum of Squares** -- degrees of freedom, SSE, and MSE
           computed from residuals after the first ``max(p, d, q)`` observations.
        4. **Ljung-Box Chi-Square Statistics** -- autocorrelation diagnostic
           at lags 12, 24, 36, and 48 (as many as the series length allows).

        .. warning::
            This method sets ``np.set_printoptions`` and
            ``pd.options.display.precision`` as a side effect.

        Parameters
        ----------
        results : ARIMAResultsWrapper
            A fitted ARIMA model, typically obtained from
            ``Models.ARIMA(...)``.
        """

        # the first thingk to do is to reverify the instance type
        checker = isinstance(results, statsmodels.tsa.arima.model.ARIMAResultsWrapper)
        if not checker:
            print("The type of the results object is not supported")
            return None

        # Set the precision of the output
        np.set_printoptions(precision=5, suppress=True)
        pd.options.display.precision = 5

        # Extract information from the result object
        terms = results.param_names #list(str)
        coefficients = results.params #series
        std_errors = results.bse #series
        t_values = results.tvalues #series
        p_values = results.pvalues
        n_coefficients = len(coefficients) - 1 #because models give an additional information on sigma^2 in the list of coefficients


        # get the order of the model and flags white noise processes
        n_model = results.nobs
        order_model = results.specification['order']
        ar_order = order_model[0]
        diff_order = order_model[1]
        ma_order = order_model[2]

        # flagging if the fitted ARIMA is not a white noise process
        order_model_flag = sum(order_model) > 0
        max_order=np.max(order_model)

        # get seasonal order vector (if SARIMA fitted instead of ARIMA) and flags models with seasonality
        so_model = results.specification['seasonal_order']
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

        # AR constant coefficient adjustment:
        if (ar_order != 0) & (diff_order == 0) & (ma_order == 0):
            AR_coefficients = coefficients[1 : ar_order + 1]
            ARIMA_constant = coefficients.iloc[0]
            adjusted_constant = ARIMA_constant * (1 - AR_coefficients.sum())
            coefficients.iloc[0] = adjusted_constant

        df_coefficients = pd.DataFrame({'Term': terms[0:n_coefficients], 'Coef': coefficients[0:n_coefficients], 'SE Coef': std_errors[0:n_coefficients], 'T-Value': t_values[0:n_coefficients], 'P-Value': p_values[0:n_coefficients]})
        # Apply same formatting style as regression
        print(df_coefficients.to_string(index=False, formatters={
            'Coef': Summary.fmt_float,
            'SE Coef': Summary.fmt_float,
            'T-Value': Summary.fmt_float,
            'P-Value': Summary.fmt_p
        }))


        # Print the ANOVA table
        print("\nRESIDUAL SUM OF SQUARES")
        print("-------------------------")
        # make a dataframe with the column names and no data
        df_rss = pd.DataFrame(columns=['DF', 'SS', 'MS'])
        # add the rows of data
        SSE = np.sum(results.resid[max_order:]**2)

        df_rss.loc[0] = [df_model, SSE, SSE/df_model]
        print(df_rss.to_string(index=False, formatters={
            'DF': lambda x: f"{int(x)}" if pd.notna(x) and x != "" else x,
            'SS': Summary.fmt_float,
            'MS': Summary.fmt_float
        }))


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
        print(df_LBtest.to_string(index=False, formatters={
            'Lag': lambda x: f"{int(x)}" if pd.notna(x) and x != "" else x,
            'Chi-Square': Summary.fmt_float,
            'P-Value': Summary.fmt_p
        }))


class Models:
    """Factory class for fitting time-series models.

    Provides static methods that wrap statsmodels estimators with
    pre-processing (e.g. manual differencing) and post-processing
    (e.g. correcting residuals) to produce results compatible with
    the :class:`Summary` printer.
    """

    @staticmethod
    def ARIMA(x, order=(0,0,0), add_constant=True):

        """Fit an ARIMA(p, d, q) differencing handled automatically

        Differences the series ``d`` times, then fits an ARMA(p, q) via
        statsmodels. After fitting, the results object is patched so that:
        - ``results.model.order`` reflects the original (p, d, q).
        - The first ``max(p, d, q)`` residuals and fitted values are set
          to NaN (they are unreliable due to differencing / burn-in).

        Parameters
        ----------
        x : pd.Series
            The time series to model.
        order : tuple of (int, int, int)
            ARIMA order as ``(p, d, q)`` where *p* is the autoregressive
            order, *d* is the differencing order, and *q* is the
            moving-average order.
        add_constant : bool
            If True, include a constant (intercept / drift) term in the
            model. If False, fit without a constant.

        Returns
        -------
        ARIMAResultsWrapper
            The fitted model results object (with patched order, residuals,
            and fitted values).
        """

        # first check if the order is a tuple or not
        if not isinstance(order, tuple) or len(order) != 3:
            raise TypeError("Order must be a tuple of length 3, e.g. (0,0,0)")

        # fit the orders into autoregressive, integration, and moving average orders
        p=order[0]
        d=order[1]
        q=order[2]

        if add_constant:
            const_coeff='c'
        else:
            const_coeff='n'

        results = arimafromlib(x, order=(p,d,q), trend=const_coeff).fit()

        return results


class StepwiseRegression:
    """Stepwise variable selection for OLS regression.

    Iteratively adds and/or removes predictors from a linear regression
    model based on p-value thresholds, printing intermediate results at
    each step. Supports three selection strategies via the ``direction``
    parameter: forward-only, backward-only, or both (forward then backward
    at each step).

    The workflow follows a scikit-learn-like pattern::

        model = StepwiseRegression(direction='both', alpha_to_enter=0.15)
        model.fit(y, X)
        # final fitted OLS results available at model.model_fit

    Attributes
    ----------
    model_fit : RegressionResultsWrapper or None
        The OLS results object after fitting. None before :meth:`fit`
        is called.
    variables_to_include : list of int
        Column indices (into ``X``) of the predictors currently in the model.
    """

    def __init__(self, add_constant = True, direction = 'both', alpha_to_enter = 0.15, alpha_to_remove = 0.15, max_iterations = 100):
        """Initialize the stepwise regression configuration.

        Parameters
        ----------
        add_constant : bool, optional
            If True, an intercept column is added to the design matrix
            at each step. Default is True.
        direction : {'both', 'forward', 'backward'}, optional
            Selection strategy. ``'forward'`` adds one variable per step,
            ``'backward'`` removes variables per step, ``'both'`` performs
            a forward step followed by a backward step each iteration.
            Default is ``'both'``.
        alpha_to_enter : float, optional
            Maximum p-value for a predictor to be added during forward
            selection. Default is 0.15.
        alpha_to_remove : float, optional
            Minimum p-value above which a predictor is removed during
            backward elimination. Default is 0.15.
        max_iterations : int, optional
            Safety limit on the number of forward/backward iterations.
            Default is 100.
        """
        self.add_constant = add_constant
        self.direction = direction
        self.alpha_to_enter = alpha_to_enter
        self.alpha_to_remove = alpha_to_remove
        self.max_iterations = max_iterations
        self.break_loop = False
        self.model_fit = None

    def fit(self, y, X):
        """Run the stepwise selection procedure and fit the final model.

        Iterates forward selection, backward elimination, or both (depending
        on ``self.direction``) until no more variables can be added or
        removed, or ``max_iterations`` is reached. Prints step-by-step
        progress to stdout.

        Parameters
        ----------
        y : pd.Series or array-like
            The response (dependent) variable.
        X : pd.DataFrame
            The predictor (independent) variables. Column names are used
            as term labels in the output.

        Returns
        -------
        self
            The fitted ``StepwiseRegression`` instance. The final OLS
            results are stored in ``self.model_fit``.
        """
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
        """Perform one forward selection step.

        Tries adding each predictor not yet in the model one at a time,
        fits an OLS for each candidate, and keeps the one whose p-value
        is below ``alpha_to_enter`` and is the smallest. If no candidate
        qualifies, sets ``self.break_loop = True`` to stop iteration.

        Returns
        -------
        self
        """

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

                # get the p value from the fitted regression
                pval = model_fit.pvalues.iloc[-1]

                # if the p-value of the new variable is less than the alpha_to_enter level, 
                # add the variable to the list of variables to include
                if pval < self.alpha_to_enter and pval < selected_pvalue:
                    selected_pvalue = pval
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
        """Perform one backward elimination step.

        Sorts the current model's predictor p-values in descending order
        and removes all predictors whose p-value exceeds
        ``alpha_to_remove``. If no predictor is removed, the model is
        unchanged (and ``self.break_loop`` is set to True when running
        in pure backward mode).

        Returns
        -------
        self
        """

        print('Backward Elimination')

        original_variables = self.variables_to_include

        # sort the pvalues in descending order and remove the variable with pvalue > alpha_to_remove
        if self.add_constant:
            sorted_pvalues = self.model_fit.pvalues[1:].sort_values(ascending = False)
        else:
            sorted_pvalues = self.model_fit.pvalues.sort_values(ascending = False)

        testing_variables = original_variables.copy()

        for i in range(len(sorted_pvalues)):
            if sorted_pvalues.iloc[i] > self.alpha_to_remove:
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
        """Print a compact summary of the current stepwise model.

        Outputs two sections to stdout:
        1. **Coefficients** -- term names, coefficient estimates, and p-values.
        2. **Model Summary** -- standard error of the regression (S),
           R-squared, and adjusted R-squared.

        This is a lighter version of :meth:`Summary.regression` used for
        intermediate stepwise output (no ANOVA or VIF).
        """
        # Extract information from the result object
        results = self.model_fit
        terms = results.model.exog_names
        coefficients = results.params
        p_values = results.pvalues

        print("\nCOEFFICIENTS")
        print("------------")
        df_coefficients = pd.DataFrame({'Term': terms, 'Coef': coefficients, 'P-Value': p_values})

        def fmt_float(x):
            if x == "" or not pd.notna(x):
                return ""
            try:
                ax = abs(float(x))
            except (ValueError, TypeError):
                return str(x)
            if ax >= 100:
                return f"{float(x):.0f}"
            elif ax >= 10:
                return f"{float(x):.2f}"
            elif ax >= 1:
                return f"{float(x):.3f}"
            elif ax >= 0.01:
                return f"{float(x):.4f}"
            else:
                return f"{float(x):.3e}"

        def fmt_p(x):
            if x == "" or not pd.notna(x):
                return ""
            else:
                return f"{float(x):.3f}"

        print(df_coefficients.to_string(index=False, formatters={
            'Coef': fmt_float,
            'P-Value': fmt_p
        }))

        # Print the R-squared and adjusted R-squared
        print("\nMODEL SUMMARY")
        print("-------------")
        S = np.std(results.resid, ddof=len(terms))
        df_model_summary = pd.DataFrame({'S': [S], 'R-sq': [results.rsquared], 'R-sq(adj)': [results.rsquared_adj]})
        print(df_model_summary.to_string(index=False))

class Assumptions:
    """Statistical assumption checker for normality and independence.

    Provides methods to test whether data satisfy the normality and
    independence assumptions commonly required by regression and
    time-series models. Each method prints test results to stdout and
    optionally produces diagnostic plots (Q-Q, ACF, PACF).

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, or np.ndarray
        The data to test. If a DataFrame is passed, multi-column methods
        like :meth:`all` will test each column separately. numpy arrays
        are automatically converted to ``pd.Series`` with a warning.
        NaN values are dropped on initialization.

    Attributes
    ----------
    data : pd.Series or pd.DataFrame
        The stored data with NaN values removed.
    """

    def __init__(self, data):
        """Initialize the Assumptions checker.

        Parameters
        ----------
        data : pd.Series, pd.DataFrame, or np.ndarray
            The data to test. numpy arrays are converted to ``pd.Series``
            with a warning. NaN values are dropped automatically.
        """
        if isinstance(data, np.ndarray):
            warnings.warn(
                "A numpy array was passed to the Assumptions class and converted to pd.Series.\n"
                "Note that all other methods in qda-toolkit only accept pd.Series and pd.DataFrame.",
                UserWarning
            )
            data = pd.Series(data)

        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                data = data.squeeze()

        self.data = data.dropna()

    def normality(self, qqplot=True, test='shapiro-wilk'):

        """Test whether the data follow a normal distribution.

        Runs a formal hypothesis test for normality and optionally displays
        a Q-Q plot for visual inspection. The null hypothesis is that the
        data are normally distributed.

        Parameters
        ----------
        qqplot : bool, optional, default = True
            If True, display a Q-Q (quantile-quantile) plot comparing the
            data quantiles against a theoretical normal distribution.

        test : {'shapiro-wilk', 'anderson-darling'}, optional. default = 'shapiro-wilk'
            Which normality test to perform
            For the Anderson-Darling test, the p-value is approximated
            using the D'Agostino & Stephens (1986) formulas.

        Returns
        -------
        stat : float
            The test statistic (W for Shapiro-Wilk, A² for Anderson-Darling).
        p_value : float
            The p-value of the test. Small values (e.g. < 0.05) indicate
            the data are unlikely to be normally distributed.
        """

        if isinstance(self.data, pd.DataFrame):
            if self.data.shape[1] > 1:
                raise TypeError("Data not supported, please specify the column name or use Assumptions(data).all() instead")

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

        """Test whether the data are independent (free of autocorrelation).

        Runs a formal hypothesis test for serial independence and optionally
        displays ACF and PACF plots. The null hypothesis is that the data
        are independent (no autocorrelation).

        Parameters
        ----------
        plotit : bool, optional
            If True, display ACF and PACF plots. Default is True.

        ac_test : {'runs', 'bartlett', 'lbq'}, optional
            Which independence test to perform. Default is ``'runs'``.

            - ``'runs'`` -- Wald-Wolfowitz runs test (non-parametric).
            - ``'bartlett'`` -- tests whether the autocorrelation at a
              specific ``lag`` is significantly different from zero using
              Bartlett's approximation.
            - ``'lbq'`` -- Ljung-Box Q test for cumulative autocorrelation
              up to a specific ``lag``.
        lag : int, optional
            The specific lag to test. **Required** when ``ac_test`` is
            ``'bartlett'`` or ``'lbq'``; ignored for ``'runs'``.
        nlags : int, optional
            Number of lags to display in the ACF/PACF plots. If None,
            defaults to ``min(len(data) // 3, 200)``.

        Returns
        -------
        stat : float or None
            The test statistic, or None if the test was not run.
        p_value : float or None
            The p-value. Small values (e.g. < 0.05) suggest the data
            exhibit significant autocorrelation.
        """

        if isinstance(self.data, pd.DataFrame):
            if self.data.shape[1] > 1:
                    raise TypeError("Data not supported, please specify the column name or use Assumptions(data).all() instead")

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
    
    def all(self, norm_test='shapiro-wilk', ac_test='runs', lag=None, nlags=None, plotit=True):

        """Run both normality and independence tests on every column at once.

        For each column in ``self.data``, performs the specified normality
        and independence tests, collects the p-values into a summary
        DataFrame, and optionally generates a grid of Q-Q, ACF, and PACF
        plots (one column of plots per data column).

        Parameters
        ----------
        norm_test : {'shapiro-wilk', 'anderson-darling'}, optional
            Normality test to use. Default is ``'shapiro-wilk'``.
        ac_test : {'runs', 'bartlett', 'lbq'}, optional
            Independence test to use. Default is ``'runs'``.
        lag : int, optional
            Lag for ``'bartlett'`` or ``'lbq'`` tests. Required if
            ``ac_test`` is not ``'runs'``.
        nlags : int, optional
            Number of lags for the ACF/PACF plots. If None, defaults to
            ``min(len(data) // 3, 200)``.
        plotit : bool, optional
            If True, display a grid of Q-Q, ACF, and PACF plots for
            every column. Default is True.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns matching ``self.data.columns`` and
            two rows: one for the normality test p-value and one for the
            independence test p-value.
        """

        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data not supported, please enter a pd.DataFrame with n > 1 columns")
        
        else:

            # get how many columns the data has
            if isinstance(self.data, pd.DataFrame):
                n_cols = self.data.shape[1]
            else:
                n_cols = 1 # if the data is a Series, it has only one column

            if nlags is None:
                nlags = min(len(self.data) // 3, 200)
            else:
                # check if the number of lags is less than the length of the data
                if nlags > len(self.data):
                    raise ValueError("The number of lags must be less than the length of the data.")
                
            assumptions_results = pd.DataFrame(columns=self.data.columns, index=[norm_test+'test P-Value', ac_test+' test P-Value'])
            fig, axes = plt.subplots(3, n_cols, figsize=(12, 5 * n_cols))
            for i, col in enumerate(self.data.columns):
                
                if norm_test == 'shapiro-wilk':
                    _, p_value_norm = stats.shapiro(self.data[col])
                elif norm_test == 'anderson-darling':
                    result = stats.anderson(self.data[col], dist='norm')
                    stat = result.statistic
                    if stat >= 0.6:
                        p_value_norm = np.exp(1.2937 - 5.709 * stat + 0.0186 * (stat ** 2))
                    elif stat >= 0.34:
                        p_value_norm = np.exp(0.9177 - 4.279 * stat - 1.38 * (stat ** 2))
                    elif stat >= 0.2:
                        p_value_norm = 1 - np.exp(-8.318 + 42.796 * stat - 59.938 * (stat ** 2))
                    else:
                        p_value_norm = 1 - np.exp(-13.436 + 101.14 * stat - 223.73 * (stat ** 2))
                else:
                    raise ValueError("Invalid normality test type. Choose 'shapiro-wilk' or 'anderson-darling'.")
                
                acf_values, stat_lbq, _ = acf(self.data[col], nlags = nlags, qstat=True, fft = False)

                # check if the lag is specified for the Bartlett or LBQ test
                if ac_test in ['bartlett', 'lbq'] and lag is None:
                    raise ValueError("The lag must be specified for the Bartlett or LBQ test.")

                if ac_test == 'runs':
                    stat, p_value_indep = runstest_1samp(self.data[col], correction=False)

                elif ac_test == 'bartlett':
                    rk = acf_values[lag]
                    stat = rk
                    p_value_indep = 2 * (1 - stats.norm.cdf(abs(stat) * np.sqrt(len(self.data[col]))))
                    
                elif ac_test == 'lbq':
                    stat = stat_lbq[lag - 1]
                    p_value_indep = 1 - stats.chi2.cdf(stat, lag)

                assumptions_results.loc[norm_test + ' test P-Value', col] = p_value_norm
                assumptions_results.loc[ac_test + ' test P-Value', col] = p_value_indep

                if plotit:
                    # Q-Q plot
                    stats.probplot(self.data[col], dist="norm", plot=axes[0, i])
                    axes[0, i].set_title(f'Q-Q Plot for {col}')

                    # ACF plot
                    sgt.plot_acf(self.data[col], lags=nlags, zero=False, ax=axes[1, i])
                    axes[1, i].set_title(f'ACF Plot for {col}')
                    axes[1, i].set_ylim(-1, 1)

                    # PACF plot
                    sgt.plot_pacf(self.data[col], lags=nlags, zero=False, ax=axes[2, i], method='ywm')
                    axes[2, i].set_title(f'PACF Plot for {col}')
                    axes[2, i].set_ylim(-1, 1)

            print(assumptions_results)
            plt.tight_layout()
            plt.show()
            
            return assumptions_results