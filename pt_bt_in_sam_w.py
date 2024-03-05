import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, cross


def prep_data(tickers):
    data = pd.read_csv('snp500_daily_prices_volume20-23_reco.csv', index_col='Date', parse_dates=True,
                       date_format='%d/%m/%Y')

    if any(data[data['tic'] == tickers[0]].loc[:, 'Close']
           - data[data['tic'] == tickers[1]].loc[:, 'Close'] < 0):
        data_sprd = data[data['tic'] == tickers[1]].loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']][:506] - \
                    data[data['tic'] == tickers[0]].loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']][:506]
    else:
        data_sprd = data[data['tic'] == tickers[0]].loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']][:506] - \
                    data[data['tic'] == tickers[1]].loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']][:506]

    data_sprd['mean_sprd'] = data_sprd['Close'].mean()
    data_sprd['stdv_sprd'] = data_sprd['Close'].std()

    return data_sprd


pairs = [
         ('CLF', 'X'),
         ('IEX', 'ITW'),
         ('CDNS', 'SNPS'),
         ('EL', 'ETN'),
         ('AME', 'NSC'),
         ('SLB', 'SLG'),
         ('AN', 'URI'),
         ('BXP', 'KMI'),
         ('AVB', 'UDR'),
         ('ELV', 'UNH'),
         ('CMA', 'FITB'),
         ('BA', 'UAL'),
         ('NCR', 'PVH'),
         ('UNM', 'VTR'),
         ('FRT', 'REG'),
         # ('L', 'USB')
]

params = [
          (3.0, 4.0, 0.1, 45),  # 'CLF', 'X'
          (2.5, 3.5, 0.1, 30),  # 'IEX', 'ITW'
          (1.5, 2.5, 0.1, 55),  # 'CDNS', 'SNPS'
          (1.5, 2.5, 0.1, 35),  # 'EL', 'ETN'
          (2.5, 3.0, 0.1, 35),  # 'AME', 'NSC'
          (2.5, 3.0, 0.1, 50),  # 'SLB', 'SLG'
          (2.0, 2.5, 0.1, 60),  # 'AN', 'URI'
          (1.5, 2.0, 0.1, 30),  # 'BXP', 'KMI'
          (1.5, 2.5, 0.6, 35),  # 'AVB', 'UDR'
          (2.5, 3.0, 0.1, 30),  # 'ELV', 'UNH'
          (1.0, 2.0, 0.4, 50),  # 'CMA', 'FITB'
          (2.5, 3.0, 0.1, 30),  # 'BA', 'UAL'
          (1.5, 2.0, 0.1, 60),  # 'NCR', 'PVH'
          (1.5, 2.0, 0.6, 30),  # 'UNM', 'VTR'
          (1.5, 2.5, 0.1, 30),  # 'FRT', 'REG'
          # (3.0, 3.5, 0.1, 30),  # 'L', 'USB'
          ]

# best_risk_adj_weights
weights = [0.01764442668490671, 0.1552009877612588, 0.13093673191044053,
           0.011113027453893892, 0.08559107871160328, 0.04039152636223379,
           0.050810270972562556, 0.00684237318300828, 0.13319658806292362,
           0.04282124769852785, 0.036653297463907795, 0.08178267190918927,
           0.06386118361620081, 0.043466738498820885, 0.09968784971052214]

# # min_variance_weights
# weights = [0.07310862818339872, 0.13698067073049852, 0.024534534955732924,
#            0.0021674030469537907, 0.13126415001743624, 0.15159276799301297,
#            0.038973066656028184, 0.0067487697965277, 0.026393465523456283,
#            0.09331023247588745, 0.02914584223134995, 0.011647465828700744,
#            0.001308290372887194, 0.15322324531513978, 0.1196014668729895]

for j, i in enumerate(pairs):

    class PairsTradingStrategy(Strategy):
        nstdv, stop_loss, size, hold_days = params[j]
        exit = 0.0

        def init(self):
            super().init()

            self.last_trade_date = self.data.index[0]

            def z_score_func(c):
                return (c - self.data['mean_sprd']) / self.data['stdv_sprd']

            self.zsi = self.I(z_score_func, self.data.Close)

        def next(self):
            super().next()

            if self.stop_loss > self.zsi > self.nstdv:
                self.sell(size=self.size)
            elif -self.stop_loss < self.zsi < -self.nstdv:
                self.buy(size=self.size)
            elif cross(self.zsi, self.exit) or \
                    crossover(self.zsi, self.stop_loss) or \
                    crossover(-self.stop_loss, self.zsi) or \
                    (self.data.index[-1] - self.last_trade_date) >= pd.Timedelta(days=self.hold_days):
                self.position.close()

            if len(self.trades) != 0:
                self.last_trade_date = self.trades[-1].entry_time


    aum = 10000000

    dataf = prep_data(i)

    bt = Backtest(dataf, PairsTradingStrategy, commission=.002,
                  cash=(aum * weights[j]), trade_on_close=True)

    stats = bt.run()
    print(stats)

    # opt = bt.optimize(nstdv=list(np.arange(3.0, 4.5, 0.5)),
    #                   stop_loss=list(np.arange(3.5, 6.5, 0.5)),
    #                   constraint=lambda x: x.nstdv < x.stop_loss,
    #                   size=list(np.arange(0.1, 1.00, 0.1)),
    #                   hold_days=list(np.arange(30, 65, 5)),
    #                   maximize='Sharpe Ratio')
    #
    # opt_params = opt._strategy
    # print(opt_params)
    # print(opt)
    returns_stream = stats._equity_curve
    column = str(pairs[j][0]) + "_" + str(pairs[j][1])
    pnl = pd.DataFrame({column: pd.Series(returns_stream['Equity'].pct_change())}).transpose()
    pnl.to_csv(f"{column}_pnl_is.csv")
    stats.to_csv(f"{column}_stats_is.csv")
    returns_stream.to_csv(f"{column}_return_stream.csv")

    bt.plot(filename=f"{column}_is.html")

