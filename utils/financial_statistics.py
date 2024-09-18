import pandas as pd


class FinancialStatistics:
    DAYS_PER_YEAR: int = 261 # 365 - weekends

    @staticmethod
    def annualise_daily_ret(ret: float, N: int) -> float:
        return (1 + ret) ** (FinancialStatistics.DAYS_PER_YEAR / N) - 1

    @staticmethod
    def annualised_return(ts: pd.Series) -> float:
        ret = (1+ts).prod() - 1
        N: int = len(ts)
        return FinancialStatistics.annualise_daily_ret(ret=ret, N=N)

    @staticmethod
    def annualised_vol(ts: pd.Series) -> float:
        vol = ts.std()
        return vol * (FinancialStatistics.DAYS_PER_YEAR ** 0.5)

    @staticmethod
    def sharpe_ratio(ts: pd.Series) -> float:
        """Sharpe Ratio, assuming a risk-free-rate of 0."""
        ann_ret: float = FinancialStatistics.annualised_return(ts)
        ann_vol: float = FinancialStatistics.annualised_vol(ts)
        return ann_ret / ann_vol

    @staticmethod
    def max_drawdown(ts: pd.Series) -> float:
        cumulative = (1+ts).cumprod()
        rolling_max = cumulative.cummax()
        daily_drawdowns = (cumulative / rolling_max) - 1
        max_drawdown = daily_drawdowns.cummin()
        return max_drawdown.values[-1]

    @staticmethod
    def VaR(ts: pd.Series, p = 0.05) -> float:
        """Empirical VaR"""
        ts_arr = ts.dropna().to_numpy()
        ts_arr.sort()
        N = len(ts_arr)
        quantile = ts_arr[int(N * p)]
        # if annualise:
        #     return FinancialStatistics.annualise_daily_ret(ret=quantile, N=1)
        return quantile
    
    @staticmethod
    def CVaR(ts: pd.Series, p = 0.05) -> float:
        """Empirical CVaR"""
        VaR: float = FinancialStatistics.VaR(ts=ts, p=p)
        CVaR: float = ts[ts<=VaR].mean()
        # if annualise:
        #     return FinancialStatistics.annualise_daily_ret(ret=CVaR, N=1)
        return CVaR
