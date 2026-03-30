import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import scipy.integrate as spi
import scipy.special as sps


class ControlCharts:
    """Statistical process control (SPC) charts for monitoring process stability.

    Provides static methods for constructing common control charts:
    Individual (I), Individual-Moving Range (IMR), Xbar-R, Xbar-S,
    CUSUM, EWMA, Hotelling T-squared, and P charts. Each method computes
    control limits, flags out-of-control points (TEST1), and optionally
    plots the chart.

    All chart methods follow a consistent pattern: they accept a
    DataFrame, compute control limits from a calibration subset, append
    the limits and alarm columns to a copy of the DataFrame, and return it.
    """

    @staticmethod
    def I(original_df, col_name, K = 3, subset_size = None, plotit = True):
        """Construct an Individual (I) control chart for single observations.

        Computes control limits using the average moving range (MR) with
        the unbiasing constant d2 for n=2. Points outside UCL/LCL are
        flagged in the ``I_TEST1`` column.

        Parameters
        ----------
        original_df : pd.DataFrame
            DataFrame containing the process data.
        col_name : str
            Name of the column to chart.
        K : int or float, optional
            Number of standard deviations for the control limits.
            Default is 3 (standard 3-sigma limits).
        subset_size : int, optional
            Number of initial rows used to estimate the center line and
            control limits (Phase I calibration). If None, all rows are
            used. A vertical dashed line is drawn at the boundary when
            ``subset_size < len(original_df)``.
        plotit : bool, optional
            If True, display the I chart. Default is True.

        Returns
        -------
        pd.DataFrame
            A copy of ``original_df`` with added columns:
            ``MR``, ``I_UCL``, ``I_CL``, ``I_LCL``, ``I_TEST1``.
        """
        # Check if df is a pandas DataFrame
        if not isinstance(original_df, pd.DataFrame):
            raise TypeError('The data must be a pandas DataFrame.')

        # Check if the col_name exists in the DataFrame
        if col_name not in original_df.columns:
            raise ValueError('The column name does not exist in the DataFrame.')

        # get the size of the DataFrame
        n = 2
        d2 = constants.getd2(n)
        D4 = constants.getD4(n, K)

        if subset_size is None:
            subset_size = len(original_df)
        elif subset_size > len(original_df):
            raise ValueError('The subset size must be less than the number of rows in the DataFrame.')

        # Create a copy of the original DataFrame
        df = original_df.copy()
        
        # Calculate the moving range
        df['MR'] = df[col_name].diff().abs()
        # Create columns for the upper and lower control limits
        df['I_UCL'] = df[col_name].iloc[:subset_size].mean() + (K*df['MR'].iloc[:subset_size].mean()/d2)
        df['I_CL'] = df[col_name].iloc[:subset_size].mean()
        df['I_LCL'] = df[col_name].iloc[:subset_size].mean() - (K*df['MR'].iloc[:subset_size].mean()/d2)
        # Define columns for the Western Electric alarm rules
        df['I_TEST1'] = np.where((df[col_name] > df['I_UCL']) | (df[col_name] < df['I_LCL']), df[col_name], np.nan)

        if plotit == True:
            # Plot the I and MR charts
            plt.title(('I chart of %s' % col_name))
            plt.plot(df[col_name], color='mediumblue', linestyle='--', marker='o')
            plt.plot(df['I_UCL'], color='firebrick', linewidth=1)
            plt.plot(df['I_CL'], color='g', linewidth=1)
            plt.plot(df['I_LCL'], color='firebrick', linewidth=1)
            plt.ylabel('Individual Value')
            plt.xlabel('Sample number')
            # add the values of the control limits on the right side of the plot
            plt.text(len(df)+.5, df['I_UCL'].iloc[0], 'UCL = {:.3f}'.format(df['I_UCL'].iloc[0]), verticalalignment='center')
            plt.text(len(df)+.5, df['I_CL'].iloc[0], 'CL = {:.3f}'.format(df['I_CL'].iloc[0]), verticalalignment='center')
            plt.text(len(df)+.5, df['I_LCL'].iloc[0], 'LCL = {:.3f}'.format(df['I_LCL'].iloc[0]), verticalalignment='center')
            # highlight the points that violate the alarm rules
            plt.plot(df['I_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
            # set the x-axis limits
            plt.xlim(-1, len(df))

            if subset_size < len(original_df):
                plt.vlines(subset_size-.5, df['I_LCL'].iloc[0], df['I_UCL'].iloc[0], color='k', linestyle='--')

            plt.tight_layout()
            plt.show()

        return df

    @staticmethod
    def IMR(original_df, col_name, K = 3, subset_size = None, run_rules = False, plotit = True):
        """Construct an Individual-Moving Range (I-MR) control chart.

        Plots both an Individual chart (top) and a Moving Range chart
        (bottom). Control limits are computed using the average MR with
        unbiasing constants d2 and D4 for n=2.

        Parameters
        ----------
        original_df : pd.DataFrame
            DataFrame containing the process data.
        col_name : str
            Name of the column to chart.
        K : int or float, optional
            Number of standard deviations for the control limits.
            Default is 3.
        subset_size : int, optional
            Number of initial rows used to estimate control limits
            (Phase I calibration). If None, all rows are used.
        run_rules : bool, optional
            If True, apply Western Electric run rules (Tests 2-8) in
            addition to the basic Test 1. Default is False.
            **Note:** this feature is not yet fully implemented.
        plotit : bool, optional
            If True, display the I-MR chart. Default is True.

        Returns
        -------
        pd.DataFrame
            A copy of ``original_df`` with added columns:
            ``MR``, ``I_UCL``, ``I_CL``, ``I_LCL``, ``I_TEST1``,
            ``MR_UCL``, ``MR_CL``, ``MR_LCL``, ``MR_TEST1``.
        """
        # Check if df is a pandas DataFrame
        if not isinstance(original_df, pd.DataFrame):
            raise TypeError('The data must be a pandas DataFrame.')

        # Check if the col_name exists in the DataFrame
        if col_name not in original_df.columns:
            raise ValueError('The column name does not exist in the DataFrame.')

        # get the size of the DataFrame
        n = 2
        d2 = constants.getd2(n)
        D4 = constants.getD4(n, K)

        if subset_size is None:
            subset_size = len(original_df)
        elif subset_size > len(original_df):
            raise ValueError('The subset size must be less than the number of rows in the DataFrame.')

        # Create a copy of the original DataFrame
        df = original_df.copy()
        
        # Calculate the moving range
        df['MR'] = df[col_name].diff().abs()
        # Create columns for the upper and lower control limits
        df['I_UCL'] = df[col_name].iloc[:subset_size].mean() + (K*df['MR'].iloc[:subset_size].mean()/d2)
        df['I_CL'] = df[col_name].iloc[:subset_size].mean()
        df['I_LCL'] = df[col_name].iloc[:subset_size].mean() - (K*df['MR'].iloc[:subset_size].mean()/d2)
        # Define columns for the Western Electric alarm rules
        df['I_TEST1'] = np.where((df[col_name] > df['I_UCL']) | (df[col_name] < df['I_LCL']), df[col_name], np.nan)

        # Create columns for the upper and lower control limits
        df['MR_UCL'] = D4 * df['MR'].iloc[:subset_size].mean()
        df['MR_CL'] = df['MR'].iloc[:subset_size].mean()
        df['MR_LCL'] = 0
        # Define columns for the Western Electric alarm rules
        df['MR_TEST1'] = np.where((df['MR'] > df['MR_UCL']) | (df['MR'] < df['MR_LCL']), df['MR'], np.nan)

        if plotit == True:
            # Plot the I and MR charts
            fig, ax = plt.subplots(2, 1, sharex=True)
            fig.suptitle(('I-MR charts of %s' % col_name))
            ax[0].plot(df[col_name], color='mediumblue', linestyle='--', marker='o')
            ax[0].plot(df['I_UCL'], color='firebrick', linewidth=1)
            ax[0].plot(df['I_CL'], color='g', linewidth=1)
            ax[0].plot(df['I_LCL'], color='firebrick', linewidth=1)
            ax[0].set_ylabel('Individual Value')
            ax[1].set_xlabel('Sample number')
            # add the values of the control limits on the right side of the plot
            ax[0].text(len(df)+.5, df['I_UCL'].iloc[0], 'UCL = {:.3f}'.format(df['I_UCL'].iloc[0]), verticalalignment='center')
            ax[0].text(len(df)+.5, df['I_CL'].iloc[0], 'CL = {:.3f}'.format(df['I_CL'].iloc[0]), verticalalignment='center')
            ax[0].text(len(df)+.5, df['I_LCL'].iloc[0], 'LCL = {:.3f}'.format(df['I_LCL'].iloc[0]), verticalalignment='center')
            # highlight the points that violate the alarm rules
            ax[0].plot(df['I_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
            ax[0].set_xlim(-1, len(df))

            ax[1].plot(df['MR'], color='mediumblue', linestyle='--', marker='o')
            ax[1].plot(df['MR_UCL'], color='firebrick', linewidth=1)
            ax[1].plot(df['MR_CL'], color='g', linewidth=1)
            ax[1].plot(df['MR_LCL'], color='firebrick', linewidth=1)
            ax[1].set_ylabel('Moving Range')
            ax[1].set_xlabel('Sample number')
            # add the values of the control limits on the right side of the plot
            ax[1].text(len(df)+.5, df['MR_UCL'].iloc[0], 'UCL = {:.3f}'.format(df['MR_UCL'].iloc[0]), verticalalignment='center')
            ax[1].text(len(df)+.5, df['MR_CL'].iloc[0], 'CL = {:.3f}'.format(df['MR_CL'].iloc[0]), verticalalignment='center')
            ax[1].text(len(df)+.5, df['MR_LCL'].iloc[0], 'LCL = {:.3f}'.format(df['MR_LCL'].iloc[0]), verticalalignment='center')
            # highlight the points that violate the alarm rules
            ax[1].plot(df['MR_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
            ax[1].set_xlim(-1, len(df))
            # set the x-axis limits
            ax[1].set_xlim(-1, len(df))

            if subset_size < len(original_df):
                ax[0].axvline(x=subset_size-.5, color='k', linestyle='--')
                ax[1].axvline(x=subset_size-.5, color='k', linestyle='--')

            plt.tight_layout()
            plt.show()

        '''
        TO DO - Implement the Western Electric alarm rules for the IMR chart
        
        if run_rules == True:

            # Create a new dataframe for the Run Rules
            df_rr = pd.DataFrame()

            # add columns for the RR calculation
            # RR2 count the number of consecutive points on the same side of the center line
            K_2 = 9 
            df['I_TEST2'] = np.nan
            df_rr['I_TEST2_1'] = df[col_name] - df['I_CL']
            df_rr['I_TEST2_2'] = 0
            df_rr['I_TEST2_2'][0] = 1

            # RR3 count the number of consecutive increasing or decreasing points
            K_3 = 6
            df_IMR['I_TEST3'] = np.nan
            df_rr['I_TEST3_1'] = df_IMR[col_name].diff()
            df_rr['I_TEST3_2'] = 0
            df_rr['I_TEST3_2'][0] = 1

            # RR4 count the number of consecutive points alternating up and down around the center line
            K_4 = 14
            df['I_TEST4'] = np.nan
            df_rr['I_TEST4_1'] = 0
            df_rr['I_TEST4_1'][0] = 1

            # RR5 K out of K+1 points > 2 standard deviations from the center line (same side)
            sigma_I = df['MR'].mean()/d2
            
            K_5 = 2
            df['I_TEST5'] = np.nan
            df_rr['I_TEST5_1'] = (df[col_name] - (df['I_CL'] + 2 * sigma_I)) > 0
            df_rr['I_TEST5_2'] = (df[col_name] - (df['I_CL'] - 2 * sigma_I)) < 0
            # create a moving average of the previous K_5 points
            df_rr['I_TEST5_3'] = df_rr['I_TEST5_1'].rolling(K_5+1).mean()
            df_rr['I_TEST5_4'] = df_rr['I_TEST5_2'].rolling(K_5+1).mean()

            # RR6 K out of K+1 points > 1 standard deviations from the center line (same side)
            K_6 = 4
            df['I_TEST6'] = np.nan
            df_rr['I_TEST6_1'] = (df[col_name] - (df['I_CL'] + 1 * sigma_I)) > 0
            df_rr['I_TEST6_2'] = (df[col_name] - (df['I_CL'] - 1 * sigma_I)) < 0
            # create a moving average of the previous K_5 points
            df_rr['I_TEST6_3'] = df_rr['I_TEST5_1'].rolling(K_6+1).mean()
            df_rr['I_TEST6_4'] = df_rr['I_TEST5_2'].rolling(K_6+1).mean()

            # RR7 K out of K+1 points within 1 standard deviations from the center line (either side)
            K_7 = 15
            df['I_TEST7'] = np.nan
            # create a column that returns 1 if the point is within 1 standard deviation of the center line 
            df_rr['I_TEST7_1'] = np.abs(df[col_name] - df['I_CL']) < sigma_I
            # create a moving average of the previous K_7 points
            df_rr['I_TEST7_2'] = df_rr['I_TEST7_1'].rolling(K_7+1).mean()

            # RR8 K out of K+1 points > 1 standard deviations from the center line (either side)
            K_8 = 8
            df['I_TEST8'] = np.nan
            # create a column that returns 1 if the point is > 1 standard deviation of the center line
            df_rr['I_TEST8_1'] = np.abs(df[col_name] - df['I_CL']) > sigma_I
            # create a moving average of the previous K_8 points
            df_rr['I_TEST8_2'] = df_rr['I_TEST8_1'].rolling(K_8+1).mean()



            for i in range(1, len(df_IMR)):
                # Rule 2
                if df_rr['I_TEST2_1'][i] * df_rr['I_TEST2_1'][i-1] > 0:
                    df_rr['I_TEST2_2'][i] = df_rr['I_TEST2_2'][i-1] + 1
                else:
                    df_rr['I_TEST2_2'][i] = 1

                if df_rr['I_TEST2_2'][i] >= K_2:
                    df['I_TEST2'][i] = df[col_name][i]
                    ax[0].text(i, df[col_name][i], '2', color='k', fontsize=12)

                # Rule 3
                if df_rr['I_TEST3_1'][i] * df_rr['I_TEST3_1'][i-1] > 0:
                    df_rr['I_TEST3_2'][i] = df_rr['I_TEST3_2'][i-1] + 1
                else:
                    df_rr['I_TEST3_2'][i] = 1

                if df_rr['I_TEST3_2'][i] >= K_3:
                    df['I_TEST3'][i] = df[col_name][i]
                    ax[0].text(i, df[col_name][i], '3', color='k', fontsize=12)

                # Rule 4
                if df_rr['I_TEST3_1'][i] * df_rr['I_TEST3_1'][i-1] < 0:
                    df_rr['I_TEST4_1'][i] = df_rr['I_TEST4_1'][i-1] + 1
                else:
                    df_rr['I_TEST4_1'][i] = 1

                if df_rr['I_TEST4_1'][i] >= K_4:
                    df['I_TEST4'][i] = df[col_name][i]
                    ax[0].text(i, df[col_name][i], '4', color='k', fontsize=12)

                # Rule 5
                if df_rr['I_TEST5_3'][i] >= K_5/(K_5+1):
                    df['I_TEST5'][i] = df[col_name][i]
                elif df_rr['I_TEST5_4'][i] >= K_5/(K_5+1):
                    df['I_TEST5'][i] = df[col_name][i]
                    ax[0].text(i, df[col_name][i], '5', color='k', fontsize=12)
                
                # Rule 6
                if df_rr['I_TEST6_3'][i] >= K_6/(K_6+1):
                    df['I_TEST6'][i] = df[col_name][i]
                    ax[0].text(i, df[col_name][i], '6', color='k', fontsize=12)
                elif df_rr['I_TEST6_4'][i] >= K_6/(K_6+1):
                    df['I_TEST6'][i] = df[col_name][i]
                    ax[0].text(i, df[col_name][i], '6', color='k', fontsize=12)

                # Rule 7
                if df_rr['I_TEST7_2'][i] >= K_7/(K_7+1):
                    df['I_TEST7'][i] = df[col_name][i]
                    ax[0].text(i, df[col_name][i], '7', color='k', fontsize=12)

                # Rule 8
                if df_rr['I_TEST8_2'][i] >= K_8/(K_8+1):
                    df['I_TEST8'][i] = df[col_name][i]
                    ax[0].text(i, df[col_name][i], '7', color='k', fontsize=12)
            
            # Plot the alarm rules
            ax[0].plot(df['I_TEST2'], linestyle='none', marker='X', color='orange', markersize=10)
            ax[0].plot(df['I_TEST3'], linestyle='none', marker='X', color='orange', markersize=10)
            ax[0].plot(df['I_TEST4'], linestyle='none', marker='X', color='orange', markersize=10)
            ax[0].plot(df['I_TEST5'], linestyle='none', marker='X', color='orange', markersize=10)
            ax[0].plot(df['I_TEST6'], linestyle='none', marker='X', color='orange', markersize=10)
            ax[0].plot(df['I_TEST7'], linestyle='none', marker='X', color='orange', markersize=10)
            ax[0].plot(df['I_TEST8'], linestyle='none', marker='X', color='orange', markersize=10)
        '''

        return df
    
    @staticmethod
    def XbarR(original_df, K = 3, mean = None, subset_size = None, plotit = True):
        """Construct an Xbar-R (sample mean and range) control chart.

        Each row of ``original_df`` represents one rational subgroup, with
        columns being individual measurements within the subgroup. The
        method computes sample means and ranges, then derives control
        limits using the A2 and D3/D4 constants.

        Parameters
        ----------
        original_df : pd.DataFrame
            DataFrame where each row is a subgroup and each column is a
            measurement within that subgroup. Must have at least 2 columns.
        K : int or float, optional
            Number of standard deviations for the control limits.
            Default is 3.
        mean : float, optional
            Known process mean. If None, the grand mean of sample means
            (from the calibration subset) is used.
        subset_size : int, optional
            Number of initial rows used to estimate control limits
            (Phase I calibration). If None, all rows are used.
        plotit : bool, optional
            If True, display the Xbar and R charts. Default is True.

        Returns
        -------
        pd.DataFrame
            A copy of ``original_df`` with added columns:
            ``sample_mean``, ``sample_range``,
            ``Xbar_CL``, ``Xbar_UCL``, ``Xbar_LCL``, ``Xbar_TEST1``,
            ``R_CL``, ``R_UCL``, ``R_LCL``, ``R_TEST1``.
        """
        # get the shape of the DataFrame
        m, n = original_df.shape

        if n < 2:
            raise ValueError('The DataFrame must contain at least 2 columns.')

        # Calculate the constants
        A2 = constants.getA2(n, K)
        D4 = constants.getD4(n, K)
        D3 = constants.getD3(n, K)

        if subset_size is None:
            subset_size = len(original_df)
        elif subset_size > len(original_df):
            raise ValueError('The subset size must be less than the number of rows in the DataFrame.')

        # Create a copy of the original DataFrame
        data_XR = original_df.copy()

        # Add a column with the mean of the rows
        data_XR['sample_mean'] = original_df.mean(axis=1)
        # Add a column with the range of the rows
        data_XR['sample_range'] = original_df.max(axis=1) - original_df.min(axis=1)

        if mean is None:
            Xbar_mean = data_XR['sample_mean'].iloc[:subset_size].mean()
        else:
            Xbar_mean = mean

        R_mean = data_XR['sample_range'].iloc[:subset_size].mean()

        # Now we can compute the CL, UCL and LCL for Xbar and R
        data_XR['Xbar_CL'] = Xbar_mean
        data_XR['Xbar_UCL'] = Xbar_mean + A2 * R_mean
        data_XR['Xbar_LCL'] = Xbar_mean - A2 * R_mean

        data_XR['R_CL'] = R_mean
        data_XR['R_UCL'] = D4 * R_mean
        data_XR['R_LCL'] = D3 * R_mean

        # Define columns for the alarms
        data_XR['Xbar_TEST1'] = np.where((data_XR['sample_mean'] > data_XR['Xbar_UCL']) | 
                (data_XR['sample_mean'] < data_XR['Xbar_LCL']), data_XR['sample_mean'], np.nan)
        data_XR['R_TEST1'] = np.where((data_XR['sample_range'] > data_XR['R_UCL']) | 
                (data_XR['sample_range'] < data_XR['R_LCL']), data_XR['sample_range'], np.nan)

        if plotit:
            fig, ax = plt.subplots(2, 1, sharex=True)
            fig.suptitle(('Xbar-R charts'))
            ax[0].plot(data_XR['sample_mean'], color='mediumblue', linestyle='--', marker='o')
            ax[0].plot(data_XR['Xbar_UCL'], color='firebrick', linewidth=1)
            ax[0].plot(data_XR['Xbar_CL'], color='g', linewidth=1)
            ax[0].plot(data_XR['Xbar_LCL'], color='firebrick', linewidth=1)
            ax[0].set_ylabel('Sample Mean')
            # add the values of the control limits on the right side of the plot
            ax[0].text(len(data_XR)+.5, data_XR['Xbar_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XR['Xbar_UCL'].iloc[0]), verticalalignment='center')
            ax[0].text(len(data_XR)+.5, data_XR['Xbar_CL'].iloc[0], 'CL = {:.3f}'.format(data_XR['Xbar_CL'].iloc[0]), verticalalignment='center')
            ax[0].text(len(data_XR)+.5, data_XR['Xbar_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XR['Xbar_LCL'].iloc[0]), verticalalignment='center')
            # highlight the points that violate the alarm rules
            ax[0].plot(data_XR['Xbar_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)

            ax[1].plot(data_XR['sample_range'], color='mediumblue', linestyle='--', marker='o')
            ax[1].plot(data_XR['R_UCL'], color='firebrick', linewidth=1)
            ax[1].plot(data_XR['R_CL'], color='g', linewidth=1)
            ax[1].plot(data_XR['R_LCL'], color='firebrick', linewidth=1)
            ax[1].set_ylabel('Sample Range')
            ax[1].set_xlabel('Sample Number')
            # add the values of the control limits on the right side of the plot
            ax[1].text(len(data_XR)+.5, data_XR['R_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XR['R_UCL'].iloc[0]), verticalalignment='center')
            ax[1].text(len(data_XR)+.5, data_XR['R_CL'].iloc[0], 'CL = {:.3f}'.format(data_XR['R_CL'].iloc[0]), verticalalignment='center')
            ax[1].text(len(data_XR)+.5, data_XR['R_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XR['R_LCL'].iloc[0]), verticalalignment='center')
            # highlight the points that violate the alarm rules
            ax[1].plot(data_XR['R_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
            # set the x-axis limits
            ax[1].set_xlim(-1, len(data_XR))

            if subset_size < len(original_df):
                ax[0].axvline(x=subset_size-.5, color='k', linestyle='--')
                ax[1].axvline(x=subset_size-.5, color='k', linestyle='--')

            plt.tight_layout()
            plt.show()

        return data_XR
    
    @staticmethod
    def XbarS(original_df, K = 3, mean = None, sigma = None, subset_size = None, plotit = True):
        """Construct an Xbar-S (sample mean and standard deviation) control chart.

        Each row of ``original_df`` represents one rational subgroup.
        Control limits for the Xbar chart use the A3 constant, and limits
        for the S chart use B3/B4 (estimated sigma) or B5/B6 (known sigma).

        Parameters
        ----------
        original_df : pd.DataFrame
            DataFrame where each row is a subgroup and each column is a
            measurement within that subgroup. Must have at least 2 columns.
        K : int or float, optional
            Number of standard deviations for the control limits.
            Default is 3.
        mean : float, optional
            Known process mean. If None, the grand mean of sample means
            (from the calibration subset) is used.
        sigma : float, optional
            Known process standard deviation. If None, the average sample
            standard deviation (Sbar) is used and limits are computed with
            B3/B4 constants. If provided, limits use B5/B6 constants.
        subset_size : int, optional
            Number of initial rows used to estimate control limits
            (Phase I calibration). If None, all rows are used.
        plotit : bool, optional
            If True, display the Xbar and S charts. Default is True.

        Returns
        -------
        pd.DataFrame
            A copy of ``original_df`` with added columns:
            ``sample_mean``, ``sample_std``,
            ``Xbar_CL``, ``Xbar_UCL``, ``Xbar_LCL``, ``Xbar_TEST1``,
            ``S_CL``, ``S_UCL``, ``S_LCL``, ``S_TEST1``.
        """
        # get the shape of the DataFrame
        m, n = original_df.shape

        if n < 2:
            raise ValueError('The DataFrame must contain at least 2 columns.')
        
        # Choose the data subset to be used for the control limits
        if subset_size is None:
            subset_size = len(original_df)
        elif subset_size > len(original_df):
            raise ValueError('The subset size must be less than the number of rows in the DataFrame.')

        # Calculate the constants
        if sigma is None:
            A3 = K * 1 / (constants.getc4(n) * np.sqrt(n))
            B3 = np.maximum(1 - K * (np.sqrt(1-constants.getc4(n)**2)) / (constants.getc4(n)), 0)
            B4 = 1 + K * (np.sqrt(1-constants.getc4(n)**2)) / (constants.getc4(n))
        else:
            A3 = K * 1 / np.sqrt(n)
            B5 = np.maximum(constants.getc4(n) - K * np.sqrt(1-constants.getc4(n)**2), 0)
            B6 = constants.getc4(n) + K * np.sqrt(1-constants.getc4(n)**2)

        # Create a copy of the original DataFrame
        data_XS = original_df.copy()

        # Add a column with the mean of the rows
        data_XS['sample_mean'] = original_df.mean(axis=1)
        # Add a column with the stdev of the rows
        data_XS['sample_std'] = original_df.std(axis=1)

        if mean is None:
            Xbar_mean = data_XS['sample_mean'].iloc[:subset_size].mean()
        else:
            Xbar_mean = mean
        
        if sigma is None:
            S_mean = data_XS['sample_std'].iloc[:subset_size].mean()
        else:
            S_mean = sigma


        # Now we can compute the CL, UCL and LCL for Xbar and S
        data_XS['Xbar_CL'] = Xbar_mean
        data_XS['Xbar_UCL'] = Xbar_mean + A3 * S_mean
        data_XS['Xbar_LCL'] = Xbar_mean - A3 * S_mean

        if sigma is None:
            data_XS['S_CL'] = S_mean
            data_XS['S_UCL'] = B4 * S_mean
            data_XS['S_LCL'] = B3 * S_mean
        else:
            data_XS['S_CL'] = constants.getc4(n) * sigma
            data_XS['S_UCL'] = B6 * sigma
            data_XS['S_LCL'] = B5 * sigma

        # Define columns for the alarms
        data_XS['Xbar_TEST1'] = np.where((data_XS['sample_mean'] > data_XS['Xbar_UCL']) | 
                        (data_XS['sample_mean'] < data_XS['Xbar_LCL']), data_XS['sample_mean'], np.nan)
        data_XS['S_TEST1'] = np.where((data_XS['sample_std'] > data_XS['S_UCL']) | 
                        (data_XS['sample_std'] < data_XS['S_LCL']), data_XS['sample_std'], np.nan)

        fig, ax = plt.subplots(2, 1, sharex=True)
        fig.suptitle(('Xbar-S charts'))
        ax[0].plot(data_XS['sample_mean'], color='mediumblue', linestyle='--', marker='o')
        ax[0].plot(data_XS['Xbar_UCL'], color='firebrick', linewidth=1)
        ax[0].plot(data_XS['Xbar_CL'], color='g', linewidth=1)
        ax[0].plot(data_XS['Xbar_LCL'], color='firebrick', linewidth=1)
        ax[0].set_ylabel('Sample Mean')
        # add the values of the control limits on the right side of the plot
        ax[0].text(len(data_XS)+.5, data_XS['Xbar_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XS['Xbar_UCL'].iloc[0]), verticalalignment='center')
        ax[0].text(len(data_XS)+.5, data_XS['Xbar_CL'].iloc[0], 'CL = {:.3f}'.format(data_XS['Xbar_CL'].iloc[0]), verticalalignment='center')
        ax[0].text(len(data_XS)+.5, data_XS['Xbar_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XS['Xbar_LCL'].iloc[0]), verticalalignment='center')
        # highlight the points that violate the alarm rules
        ax[0].plot(data_XS['Xbar_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)

        ax[1].plot(data_XS['sample_std'], color='mediumblue', linestyle='--', marker='o')
        ax[1].plot(data_XS['S_UCL'], color='firebrick', linewidth=1)
        ax[1].plot(data_XS['S_CL'], color='g', linewidth=1)
        ax[1].plot(data_XS['S_LCL'], color='firebrick', linewidth=1)
        ax[1].set_ylabel('Sample StDev')
        ax[1].set_xlabel('Sample Number')
        # add the values of the control limits on the right side of the plot
        ax[1].text(len(data_XS)+.5, data_XS['S_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XS['S_UCL'].iloc[0]), verticalalignment='center')
        ax[1].text(len(data_XS)+.5, data_XS['S_CL'].iloc[0], 'CL = {:.3f}'.format(data_XS['S_CL'].iloc[0]), verticalalignment='center')
        ax[1].text(len(data_XS)+.5, data_XS['S_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XS['S_LCL'].iloc[0]), verticalalignment='center')
        # highlight the points that violate the alarm rules
        ax[1].plot(data_XS['S_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
        # set the x-axis limits
        ax[1].set_xlim(-1, len(data_XS))

        if subset_size < len(original_df):
            ax[0].axvline(x=subset_size-.5, color='k', linestyle='--')
            ax[1].axvline(x=subset_size-.5, color='k', linestyle='--')

        plt.tight_layout()
        plt.show()

        return data_XS

    @staticmethod
    def CUSUM(data, col_name, params, mean = None, sigma_xbar = None, subset_size = None, plotit = True):
        """Construct a tabular CUSUM (Cumulative Sum) control chart.

        Tracks cumulative deviations from the target mean using upper
        (Ci+) and lower (Ci-) one-sided statistics. Signals when either
        statistic exceeds the decision interval H = h * sigma.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the process data.
        col_name : str
            Name of the column to chart.
        params : tuple of (float, float)
            CUSUM design parameters ``(h, k)`` where *h* is the decision
            interval (in sigma units) and *k* is the reference value /
            allowance (in sigma units). Common choice: ``(5, 0.5)``.
        mean : float, optional
            Known target mean. If None, estimated from the calibration
            subset.
        sigma_xbar : float, optional
            Known process standard deviation. If None, estimated as the
            sample standard deviation of the calibration subset.
        subset_size : int, optional
            Number of initial rows used to estimate mean and sigma
            (Phase I calibration). If None, all rows are used.
        plotit : bool, optional
            If True, display the CUSUM chart. Default is True.

        Returns
        -------
        pd.DataFrame
            A copy of ``data`` with added columns:
            ``Ci+``, ``Ci-``, ``Ci+_TEST1``, ``Ci-_TEST1``.
        """
        # Check if the col_name exists in the DataFrame
        if col_name not in data.columns:
            raise ValueError('The column name does not exist in the DataFrame.')

        if subset_size is None:
            subset_size = len(data)
        elif subset_size > len(data):
            raise ValueError('The subset size must be less than the number of rows in the DataFrame.')

        if sigma_xbar is None:
            sigma_xbar = data.loc[:subset_size, col_name].std()
        
        if mean is None:
            xbarbar = data.loc[:subset_size, col_name].mean()
        else:   
            xbarbar = mean


        
        h, k = params

        H = h*sigma_xbar
        K = k*sigma_xbar

        df_CUSUM = data.copy()
        df_CUSUM['Ci+'] = 0.0
        df_CUSUM['Ci-'] = 0.0

        for i in range(len(df_CUSUM)):
            if i == 0:
                df_CUSUM.loc[i, 'Ci+'] = max(0, df_CUSUM.loc[i, col_name] - (xbarbar + K))
                df_CUSUM.loc[i, 'Ci-'] = max(0, (xbarbar - K) - df_CUSUM.loc[i, col_name])
            else:
                df_CUSUM.loc[i, 'Ci+'] = max(0, df_CUSUM.loc[i, col_name] - (xbarbar + K) + df_CUSUM.loc[i-1, 'Ci+'])
                df_CUSUM.loc[i, 'Ci-'] = max(0, (xbarbar - K) - df_CUSUM.loc[i, col_name] + df_CUSUM.loc[i-1, 'Ci-'])

        df_CUSUM['Ci+_TEST1'] = np.where((df_CUSUM['Ci+'] > H) | (df_CUSUM['Ci+'] < -H), df_CUSUM['Ci+'], np.nan)
        df_CUSUM['Ci-_TEST1'] = np.where((df_CUSUM['Ci-'] > H) | (df_CUSUM['Ci-'] < -H), df_CUSUM['Ci-'], np.nan)

        if plotit == True:
            # Plot the control limits
            plt.hlines(H, 0, len(df_CUSUM), color='firebrick', linewidth=1)
            plt.hlines(0, 0, len(df_CUSUM), color='g', linewidth=1)
            plt.hlines(-H, 0, len(df_CUSUM), color='firebrick', linewidth=1)
            # Plot the chart
            plt.title('CUSUM chart of %s (h=%.2f, k=%.2f)' % (col_name, h, k))
            plt.plot(df_CUSUM['Ci+'], color='b', linestyle='-', marker='o')
            plt.plot(-df_CUSUM['Ci-'], color='b', linestyle='-', marker='D')
            # add the values of the control limits on the right side of the plot
            plt.text(len(df_CUSUM)+.5, H, 'UCL = {:.3f}'.format(H), verticalalignment='center')
            plt.text(len(df_CUSUM)+.5, 0, 'CL = {:.3f}'.format(0), verticalalignment='center')
            plt.text(len(df_CUSUM)+.5, -H, 'LCL = {:.3f}'.format(-H), verticalalignment='center')
            # highlight the points that violate the alarm rules
            plt.plot(df_CUSUM['Ci+_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
            plt.plot(-df_CUSUM['Ci-_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
            plt.xlim(-1, len(df_CUSUM))
            # add labels
            plt.xlabel('Sample')
            plt.ylabel('Cumulative Sum')

            if subset_size < len(data):
                plt.vlines(subset_size-.5, -H, H, color='k', linestyle='--')

            plt.tight_layout()
            plt.show()

        return df_CUSUM

    @staticmethod
    def EWMA(data, col_name, params, mean = None, sigma_xbar = None, subset_size = None, plotit = True):
        """Construct an EWMA (Exponentially Weighted Moving Average) control chart.

        Computes the EWMA statistic z_t = lambda * x_t + (1 - lambda) * z_{t-1}
        with time-varying 3-sigma control limits that widen from the
        center line and converge to their steady-state values.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the process data.
        col_name : str
            Name of the column to chart.
        params : float
            The smoothing parameter lambda (between 0 and 1). Smaller
            values give more weight to historical data. Common choice:
            0.05-0.25.
        mean : float, optional
            Known target mean. If None, estimated from the calibration
            subset.
        sigma_xbar : float, optional
            Known process standard deviation. If None, estimated as the
            sample standard deviation of the calibration subset.
        subset_size : int, optional
            Number of initial rows used to estimate mean and sigma
            (Phase I calibration). If None, all rows are used.
        plotit : bool, optional
            If True, display the EWMA chart. Default is True.

        Returns
        -------
        pd.DataFrame
            A copy of ``data`` with added columns:
            ``a_t`` (variance weight), ``z`` (EWMA statistic),
            ``UCL``, ``CL``, ``LCL``, ``z_TEST1``.
        """
        # Check if the col_name exists in the DataFrame
        if col_name not in data.columns:
            raise ValueError('The column name does not exist in the DataFrame.')

        if subset_size is None:
            subset_size = len(data)
        elif subset_size > len(data):
            raise ValueError('The subset size must be less than the number of rows in the DataFrame.')

        if sigma_xbar is None:
            sigma_xbar = data.loc[:subset_size, col_name].std()
        
        if mean is None:
            xbarbar = data.loc[:subset_size, col_name].mean()        
        else:
            xbarbar = mean

        lambda_ = params

        df_EWMA = data.copy()
        df_EWMA['a_t'] = lambda_/(2-lambda_) * (1 - (1-lambda_)**(2*np.arange(1, len(df_EWMA)+1)))
        
        for i in range(len(df_EWMA)):
            if i == 0:
                df_EWMA.loc[i, 'z'] = lambda_*df_EWMA.loc[i, col_name] + (1-lambda_)*xbarbar
            else:
                df_EWMA.loc[i, 'z'] = lambda_*df_EWMA.loc[i, col_name] + (1-lambda_)*df_EWMA.loc[i-1, 'z']
        
        df_EWMA['UCL'] = xbarbar + 3*sigma_xbar*np.sqrt(df_EWMA['a_t'])
        df_EWMA['CL'] = xbarbar
        df_EWMA['LCL'] = xbarbar - 3*sigma_xbar*np.sqrt(df_EWMA['a_t'])

        df_EWMA['z_TEST1'] = np.where((df_EWMA['z'] > df_EWMA['UCL']) | (df_EWMA['z'] < df_EWMA['LCL']), df_EWMA['z'], np.nan)

        if plotit == True:
            # Plot the control limits
            plt.step(df_EWMA.index, df_EWMA['UCL'], color='firebrick', linewidth=1, where='mid')
            plt.plot(df_EWMA['CL'], color='g', linewidth=1)
            plt.step(df_EWMA.index, df_EWMA['LCL'], color='firebrick', linewidth=1, where='mid')
            # Plot the chart
            plt.title('EWMA chart of %s (lambda=%.2f)' % (col_name, lambda_))
            plt.plot(df_EWMA['z'], color='b', linestyle='-', marker='o')
            # add the values of the control limits on the right side of the plot
            plt.text(len(df_EWMA)+.5, df_EWMA['UCL'].iloc[-1], 'UCL = {:.3f}'.format(df_EWMA['UCL'].iloc[-1]), verticalalignment='center')
            plt.text(len(df_EWMA)+.5, df_EWMA['CL'].iloc[-1], 'CL = {:.3f}'.format(df_EWMA['CL'].iloc[-1]), verticalalignment='center')
            plt.text(len(df_EWMA)+.5, df_EWMA['LCL'].iloc[-1], 'LCL = {:.3f}'.format(df_EWMA['LCL'].iloc[-1]), verticalalignment='center')
            # highlight the points that violate the alarm rules
            plt.plot(df_EWMA['z_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
            plt.xlim(-1, len(df_EWMA))
            # add labels
            plt.xlabel('Sample')
            plt.ylabel('EWMA')
            
            if subset_size < len(data):
                plt.vlines(subset_size-.5, df_EWMA['LCL'].iloc[-1], df_EWMA['UCL'].iloc[-1], color='k', linestyle='--')
            
            plt.tight_layout()
            plt.show()
        
        return df_EWMA
    
    @staticmethod
    def T2hotelling(original_df, col_names, sample_size, alpha, mean = None, varcov = None, plotit = True):
        """Construct a Hotelling T-squared multivariate control chart.

        Monitors multiple correlated quality characteristics simultaneously.
        The UCL distribution depends on the available information:

        - **Known mean and covariance** (both provided): chi-squared distribution.
        - **Estimated from subgroups with n > 1**: F distribution.
        - **Estimated from individual observations (n = 1)**: Beta distribution
          (Phase I) and F distribution (Phase II), using a short-range
          covariance estimator based on successive differences.

        Parameters
        ----------
        original_df : pd.DataFrame
            DataFrame containing the raw observations. The total number
            of rows must be divisible by ``n`` (subgroup size).
        col_names : list of str
            Names of the columns (quality characteristics) to monitor.
        sample_size : tuple of (int, int)
            ``(m, n)`` where *m* is the number of subgroups used for
            Phase I calibration and *n* is the number of observations
            per subgroup. Rows are split into subgroups of size *n*.
        alpha : float
            Significance level for the control limit (e.g. 0.05).
        mean : pd.Series, optional
            Known process mean vector (indexed by ``col_names``). If None,
            estimated as the grand mean of the first *m* subgroup means.
        varcov : pd.DataFrame, optional
            Known variance-covariance matrix. If None, estimated from the
            first *m* subgroups (pooled within-subgroup covariance for
            n > 1, or short-range successive-difference estimator for n = 1).
        plotit : bool, optional
            If True, display the T-squared chart. Default is True.

        Returns
        -------
        pd.DataFrame
            A DataFrame of subgroup means with added columns:
            ``T2``, ``UCL``, ``T2_TEST``.
        """

        m, n = sample_size

        # check if the dataset has the correct number of rows
        if np.mod(len(original_df), n) != 0:
            raise ValueError('The number of observations (n) must be a factor of the number of rows in the DataFrame.')

        # number of variables
        p = len(col_names)

        # create the column sample_id
        original_df['sample_id'] = np.repeat(np.arange(1, len(original_df)/n+1), n)

        # group by sample_id to calculate the mean within each sample
        sample_mean = original_df.groupby('sample_id').mean()
        

        if mean is None:
            # compute the grand mean from samples up to m
            Xbarbar = sample_mean[col_names].iloc[:m].mean()

            # reorder the columns to match the order of the columns in the DataFrame
            Xbarbar = Xbarbar.reindex(index=col_names)

        if varcov is None:
            if n > 1:
                # Compute the variance and covariance matrix of each group (sample) up to m
                original_df_subset = original_df[original_df['sample_id'] <= m]
                cov_matrix = original_df_subset.groupby('sample_id')[col_names].cov()

                # Compute the mean covariance matrix
                S = cov_matrix.groupby(level=1).mean()

                # reorder the columns to match the order of the columns in the DataFrame
                S = S.reindex(columns=col_names, index=col_names)

                # Compute the UCL for the Hotelling T2 statistic
                print('The UCL is calculated using the F distribution.')
                UCL1 = (p * (m-1) * (n-1)) / (m * (n-1) - (p-1)) * stats.f.ppf(1-alpha, p, m*n - m + 1 - p)
                UCL2 = UCL1
            else:
                # short range estimator
                # Create the V matrix using the first m samples
                V = sample_mean[col_names].iloc[:m].diff().dropna()

                # Calculate the short range estimator S2
                S = 1/2 * V.transpose().dot(V) / (m-1)

                # Compute the UCL for the Hotelling T2 statistic with the beta distribution
                print('The UCL is calculated using the BETA distribution.')
                UCL1 = ((m-1)**2)/m*stats.beta.ppf(1 - alpha, p/2, (m-p-1)/2)
                UCL2 = (p*(m+1)*(m-1))/(m*(m-p))*stats.f.ppf(1-alpha, p, m - p)

        else:
            S = varcov

            # Compute the UCL for the Hotelling T2 statistic with the chi2 distribution
            print('The UCL is calculated using the CHI2 distribution.')
            UCL1 = stats.chi2.ppf(1 - alpha, df = p)
            UCL2 = UCL1

        # Calculate the Hotelling T2 statistic for all the samples
        # Initialize the list to store the T2 values
        sample_mean['T2'] = np.nan

        # calculate the inverse of the covariance matrix
        S_inv = np.linalg.inv(S)

        # sample mean starts at 1 as index because of sample id
        # thus we need to reset index
        sample_mean.reset_index(inplace=True)

        # iterate through the length of sample_mean dataframe
        for i in range(len(sample_mean)): 
            # len is zero based indexing
            sample_mean.loc[i, "T2"] = n * (sample_mean[col_names].iloc[i]-Xbarbar).transpose().dot(S_inv).dot(sample_mean[col_names].iloc[i]-Xbarbar)

        # add the UCL to the DataFrame up to m
        sample_mean['UCL'] = UCL1
        if len(sample_mean) > m:
            sample_mean.loc[m:, 'UCL'] = UCL2


        # Add a column with the test
        sample_mean['T2_TEST'] = np.where(sample_mean['T2'] > sample_mean['UCL'], sample_mean['T2'], np.nan)

        # test
        """
        WARNING: DATA HAS NANS, ROOT CAUSED TO BE CHECKED NOW
        TODO: check operations involving sample_mean and find the NaN root cause
        """

        # Plot the Hotelling T2 statistic
        if plotit == True:
            plt.plot(sample_mean['UCL'], color='firebrick', linewidth=1)
            plt.hlines(np.median(sample_mean['T2']), 1, len(sample_mean), color='g', linewidth=1)
            plt.plot(sample_mean['T2'], color='b', linestyle='--', marker='o')
            plt.plot(sample_mean['T2_TEST'], linestyle='none', marker='s', color='firebrick', markersize=10)
            plt.title('T$^2$ chart of %s' % col_names)
            if len(sample_mean) > m and UCL1 != UCL2:
                plt.text(len(sample_mean)+1.2, UCL1, 'UCL_p1 = {:.3f}'.format(UCL1), verticalalignment='center')
                plt.text(len(sample_mean)+1.2, UCL2, 'UCL_p2 = {:.3f}'.format(UCL2), verticalalignment='center')
            else:
                plt.text(len(sample_mean)+1.2, UCL1, 'UCL = {:.3f}'.format(UCL1), verticalalignment='center')
            plt.text(len(sample_mean)+1.2, np.median(sample_mean['T2']), 'Median = {:.3f}'.format(np.median(sample_mean['T2']), verticalalignment='center'))
            plt.xlim(0, len(sample_mean)+1)
            plt.xlabel('Sample')
            plt.ylabel('T$^2$')
            plt.tight_layout()
            plt.show()

        return sample_mean

    def P(original_df, defects_col, subgroup_size, subset_size=None, known_params=None, plotit=True):
        """Construct a P (proportion defective) control chart.

        Monitors the fraction of nonconforming items per subgroup. Control
        limits are computed as pbar +/- 3*sqrt(pbar*(1-pbar)/n), so they
        may vary across samples when subgroup sizes differ.

        Parameters
        ----------
        original_df : pd.DataFrame
            DataFrame containing the process data.
        defects_col : str
            Name of the column containing the number of defective items
            per subgroup. If ``subgroup_size=1``, this column should
            contain proportions (values between 0 and 1) instead.
        subgroup_size : int or str
            Either a fixed integer subgroup size, or the name of a column
            in ``original_df`` containing per-row subgroup sizes. Set to
            1 when ``defects_col`` already contains proportions.
        subset_size : int, optional
            Number of initial rows used to estimate the mean proportion
            (Phase I calibration). If None, all rows are used.
        known_params : float, optional
            Known process proportion (pbar). If None, estimated from the
            calibration subset.
        plotit : bool, optional
            If True, display the P chart. Default is True.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with added columns:
            ``p``, ``std_dev``, ``P_CL``, ``P_UCL``, ``P_LCL``, ``P_TEST1``.
        """
        if subgroup_size is None:
            raise ValueError('The subgroup size column must be provided.')
        
        if subgroup_size == 1:
            # check if the values in defects_col are between 0 and 1
            if original_df[defects_col].min() < 0 or original_df[defects_col].max() > 1:
                raise ValueError('The values in the defects column must be between 0 and 1.')

        # If subgroup_size_col is an int and not a column name, set the subgroup size to that value
        if isinstance(subgroup_size, int):
            subgroup_size = subgroup_size
        else:
            subgroup_size = original_df[subgroup_size]
        # Calculate the proportion of defects
        original_df['p'] = original_df[defects_col] / subgroup_size

        if subset_size is None:
            subset_size = len(original_df)
        elif subset_size > len(original_df):
            raise ValueError('The subset size must be less than the number of rows in the DataFrame.')

        if known_params is not None:
            mean = known_params
        else:
            mean = original_df['p'].iloc[:subset_size].mean()

        stdev = np.sqrt((mean * (1 - mean)) / subgroup_size)

        # UCL, LCL, CL
        original_df['std_dev'] = stdev
        original_df['P_CL'] = mean
        original_df['P_UCL'] = original_df['P_CL'] + 3 * original_df['std_dev']
        original_df['P_LCL'] = original_df['P_CL'] - 3 * original_df['std_dev']
        original_df['P_LCL'] = original_df['P_LCL'].clip(lower=0)  # LCL cannot be <0
        original_df['P_TEST1'] = np.where((original_df['p'] > original_df['P_UCL']) | (original_df['p'] < original_df['P_LCL']), original_df['p'], np.nan)

        # p-Chart
        if plotit:
            plt.figure()
            plt.plot(original_df['p'], marker='o', color='blue')
            plt.step(original_df.index, original_df['P_CL'], where='mid', color='g', linestyle='-', linewidth=1)
            plt.step(original_df.index, original_df['P_UCL'], where='mid', color='firebrick', linestyle='-', linewidth=1)
            plt.step(original_df.index, original_df['P_LCL'], where='mid', color='firebrick', linestyle='-', linewidth=1)
            plt.xlabel('Sample')
            plt.ylabel('Proportion')
            plt.title('P-Chart')
            # add the values of the control limits on the right side of the plot
            plt.text(len(original_df)+.5, original_df['P_UCL'].iloc[0], 'UCL = {:.4f}'.format(original_df['P_UCL'].iloc[-1]), verticalalignment='center')
            plt.text(len(original_df)+.5, mean, r'$\bar{p}$' + ' = {:.4f}'.format(mean), verticalalignment='center')
            plt.text(len(original_df)+.5, original_df['P_LCL'].iloc[0], 'LCL = {:.4f}'.format(original_df['P_LCL'].iloc[-1]), verticalalignment='center')
            # plt.grid(True, linestyle='--', alpha=0.6)
            plt.plot(original_df['P_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
            # set the x-axis limits
            plt.xlim(-1, len(original_df))
            
            if subset_size < len(original_df):
                plt.vlines(subset_size-.5, original_df['P_LCL'].iloc[-1], original_df['P_UCL'].iloc[-1], color='k', linestyle='--')

            plt.tight_layout()
            plt.show()
            

        return original_df
    

class constants:
    """Unbiasing and scaling constants for control chart calculations.

    Computes the standard SPC constants (d2, d3, c4, A2, D3, D4) via
    numerical integration of the studentized range distribution and
    gamma function identities. All methods are static and accept the
    rational subgroup size ``n`` as input.
    """

    @staticmethod
    def getd2(n=None):
        """Compute the d2 constant (mean of the relative range).

        d2 is defined as E[W] where W follows the studentized range
        distribution with parameters ``n`` and infinite degrees of freedom.
        Used to convert the average range to an estimate of sigma:
        sigma_hat = Rbar / d2.

        Parameters
        ----------
        n : int
            Subgroup size (must be an integer >= 2).

        Returns
        -------
        float
            The d2 constant for the given subgroup size.
        """
        if n is None or n < 2 or abs(n - round(n)) != 0:
            raise ValueError("Invalid sample size ({})".format(n))
        def f(x):
            return stats.studentized_range.sf(x, n, np.inf)
        d2, _ = spi.quad(f, 0, np.inf)
        if _ > 0.001:
            print("The absolute error after numerical integration is greater than 0.001")
        return d2

    @staticmethod
    def getd3(n=None):
        """Compute the d3 constant (standard deviation of the relative range).

        d3 = sqrt(E[W^2] - d2^2), where W follows the studentized range
        distribution. Used to compute D3 and D4 constants for range chart
        control limits.

        Parameters
        ----------
        n : int
            Subgroup size (must be an integer >= 2).

        Returns
        -------
        float
            The d3 constant for the given subgroup size.
        """
        if n is None or n < 2 or abs(n - round(n)) != 0:
            raise ValueError("Invalid sample size")
        def f(x):
            return x * stats.studentized_range.sf(x, n, np.inf)
        d3, _ = spi.quad(f, 0, np.inf)
        if _ > 0.001:
            print("The absolute error after numerical integration is greater than 0.001")
        d3 = 2 * d3
        this_d2 = constants.getd2(n)
        d3 = np.sqrt(d3 - this_d2**2)
        return d3

    @staticmethod
    def getc4(n=None):
        """Compute the c4 unbiasing constant for sample standard deviation.

        c4 = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2). Used to convert
        the average sample standard deviation to an unbiased estimate of
        sigma: sigma_hat = Sbar / c4.

        Parameters
        ----------
        n : int
            Subgroup size (must be an integer >= 2).

        Returns
        -------
        float
            The c4 constant for the given subgroup size.
        """
        if n is None or n < 2 or abs(n - round(n)) != 0:
            raise ValueError("Invalid sample size")
        c4 = np.sqrt(2 / (n-1)) * (sps.gamma(n/2) / sps.gamma((n-1)/2))
        return c4

    @staticmethod
    def getA2(n=None, K = 3):
        """Compute the A2 constant for Xbar-R chart control limits.

        A2 = K / (d2 * sqrt(n)). The Xbar control limits are:
        UCL/LCL = Xbar_bar +/- A2 * Rbar.

        Parameters
        ----------
        n : int
            Subgroup size (must be an integer >= 2).
        K : int or float, optional
            Number of sigma for the control limits. Default is 3.

        Returns
        -------
        float
            The A2 constant for the given subgroup size and K.
        """
        if n is None or n < 2 or abs(n - round(n)) != 0:
            raise ValueError("Invalid sample size")
        A2 = K / (constants.getd2(n) * np.sqrt(n))
        return A2

    @staticmethod
    def getD3(n=None, K = 3):
        """Compute the D3 constant for the lower control limit of the R chart.

        D3 = max(0, 1 - K * d3 / d2). The R chart LCL = D3 * Rbar.
        For small subgroup sizes, D3 is clamped to 0 (LCL cannot be
        negative).

        Parameters
        ----------
        n : int
            Subgroup size (must be an integer >= 2).
        K : int or float, optional
            Number of sigma for the control limits. Default is 3.

        Returns
        -------
        float
            The D3 constant for the given subgroup size and K.
        """
        if n is None or n < 2 or abs(n - round(n)) != 0:
            raise ValueError("Invalid sample size")
        D3 = np.maximum(0, 1 - K * constants.getd3(n) / constants.getd2(n))
        return D3

    @staticmethod
    def getD4(n=None, K = 3):
        """Compute the D4 constant for the upper control limit of the R chart.

        D4 = 1 + K * d3 / d2. The R chart UCL = D4 * Rbar.

        Parameters
        ----------
        n : int
            Subgroup size (must be an integer >= 2).
        K : int or float, optional
            Number of sigma for the control limits. Default is 3.

        Returns
        -------
        float
            The D4 constant for the given subgroup size and K.
        """
        if n is None or n < 2 or abs(n - round(n)) != 0:
            raise ValueError("Invalid sample size")
        D4 = 1 + K * constants.getd3(n) / constants.getd2(n)
        return D4