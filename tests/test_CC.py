import numpy as np
import pandas as pd
import pytest
import qdatoolkit as qda


def test_CC_XbarR():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 5))
    data = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
    cc = qda.ControlCharts.XbarR(data, plotit=False)
    assert {'Xbar_UCL', 'Xbar_CL', 'Xbar_LCL', 'R_UCL', 'R_CL', 'R_LCL'}.issubset(cc.columns)


def test_CC_IMRR():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 5))
    data = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
    cc = qda.ControlCharts.IMRR(data, plotit=False)
    assert {'I_UCL', 'I_CL', 'I_LCL', 'MR_UCL', 'MR_CL', 'MR_LCL', 'R_UCL', 'R_CL', 'R_LCL'}.issubset(cc.columns)
    pd.testing.assert_series_equal(cc['sample_mean'], data.mean(axis=1), check_names=False)
    pd.testing.assert_series_equal(cc['MR'], data.mean(axis=1).diff().abs(), check_names=False)


def test_CC_XbarS():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 5))
    data = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
    cc = qda.ControlCharts.XbarS(data, plotit=False)
    assert {'Xbar_UCL', 'Xbar_CL', 'Xbar_LCL', 'S_UCL', 'S_CL', 'S_LCL'}.issubset(cc.columns)


def test_CC_XbarS_long_format():
    wide = pd.DataFrame([
        [9.8, 10.1, 10.0],
        [10.3, 10.5, 10.1],
        [9.7, 9.9, 10.2],
    ])
    long = pd.DataFrame({
        'sample_id': np.repeat(['a', 'b', 'c'], wide.shape[1]),
        'value': wide.to_numpy().ravel(),
    })

    expected = qda.ControlCharts.XbarS(wide, plotit=False)
    by_id = qda.ControlCharts.XbarS(
        long,
        col_name='value',
        id_column='sample_id',
        plotit=False,
    )
    by_size = qda.ControlCharts.XbarS(
        long[['value']],
        col_name='value',
        sample_size=wide.shape[1],
        plotit=False,
    )

    result_columns = [
        'sample_mean', 'sample_std', 'Xbar_UCL', 'Xbar_CL', 'Xbar_LCL',
        'S_UCL', 'S_CL', 'S_LCL',
    ]
    pd.testing.assert_frame_equal(
        by_id[result_columns], expected[result_columns], check_names=False
    )
    pd.testing.assert_frame_equal(by_size[result_columns], expected[result_columns])


def test_CC_XbarS_rejects_unequal_subgroup_sizes():
    data = pd.DataFrame({
        'sample_id': ['a', 'a', 'a', 'b', 'b'],
        'value': [1.0, 1.1, 0.9, 1.2, 1.3],
    })

    with pytest.raises(ValueError, match='same number of measurements'):
        qda.ControlCharts.XbarS(
            data,
            col_name='value',
            id_column='sample_id',
            plotit=False,
        )


def test_CC_I():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 1))
    data = pd.DataFrame(data, columns=['A'])
    cc = qda.ControlCharts.I(data, 'A', plotit=False)
    assert {'MR', 'I_UCL', 'I_CL', 'I_LCL', 'I_TEST1'}.issubset(cc.columns)


def test_CC_IMR():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 1))
    data = pd.DataFrame(data, columns=['A'])
    cc = qda.ControlCharts.IMR(data, 'A', plotit=False)
    assert {'MR', 'I_UCL', 'I_CL', 'I_LCL', 'MR_UCL', 'MR_CL', 'MR_LCL'}.issubset(cc.columns)


def test_CC_CUSUM():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 1))
    data = pd.DataFrame(data, columns=['A'])
    cc = qda.ControlCharts.CUSUM(data, 'A', h=10, k=0.5, plotit=False)
    assert {'Ci+', 'Ci-', 'Ci+_TEST1', 'Ci-_TEST1'}.issubset(cc.columns)


def test_CC_EWMA():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(20, 1))
    data = pd.DataFrame(data, columns=['A'])
    cc = qda.ControlCharts.EWMA(data, 'A', lambda_=0.2, plotit=False)
    assert {'z', 'a_t', 'UCL', 'LCL', 'z_TEST1'}.issubset(cc.columns)


def test_CC_T2hotelling():
    np.random.seed(0)
    data = np.random.normal(loc=10, scale=2, size=(30, 2))
    data = pd.DataFrame(data, columns=['A', 'B'])
    cc = qda.ControlCharts.T2hotelling(data, ['A', 'B'], (10, 3), 0.0027, plotit=False)
    assert {'T2', 'UCL', 'T2_TEST'}.issubset(cc.columns)


def test_CC_P():
    np.random.seed(0)
    subgroup_size = np.random.randint(80, 121, size=20)
    defects = np.random.binomial(subgroup_size, 0.05)
    data = pd.DataFrame({'defects': defects, 'subgroup_size': subgroup_size})
    cc = qda.ControlCharts.P(data.copy(), 'defects', 'subgroup_size', plotit=False)
    assert {'p', 'std_dev', 'P_CL', 'P_UCL', 'P_LCL', 'P_TEST1'}.issubset(cc.columns)


def test_CC_NP():
    np.random.seed(0)
    data = pd.DataFrame({'defects': np.random.binomial(100, 0.05, size=20)})
    cc = qda.ControlCharts.NP(data.copy(), 'defects', subgroup_size=100, plotit=False)
    assert {'NP_CL', 'NP_UCL', 'NP_LCL', 'NP_TEST1'}.issubset(cc.columns)


def test_CC_C():
    np.random.seed(0)
    data = pd.DataFrame({'defects': np.random.poisson(lam=4, size=20)})
    cc = qda.ControlCharts.C(data, 'defects', plotit=False)
    assert {'C_UCL', 'C_CL', 'C_LCL', 'C_TEST1'}.issubset(cc.columns)


def test_CC_U():
    np.random.seed(0)
    subgroup_size = np.random.randint(5, 16, size=20)
    defects = np.random.poisson(lam=subgroup_size * 0.4)
    data = pd.DataFrame({'defects': defects, 'subgroup_size': subgroup_size})
    cc = qda.ControlCharts.U(data, 'defects', 'subgroup_size', plotit=False)
    assert {'u', 'U_UCL', 'U_CL', 'U_LCL', 'U_TEST1'}.issubset(cc.columns)


if __name__ == '__main__':
    pass
    # test_CC_I()
    # test_CC_XbarR()
    # test_CC_XbarS()
    # test_CC_IMR()
    # test_CC_CUSUM()
    # test_CC_EWMA()
    # test_CC_T2hotelling()
    # test_CC_P()
    # test_CC_NP()
    # test_CC_C()
    # test_CC_U()
