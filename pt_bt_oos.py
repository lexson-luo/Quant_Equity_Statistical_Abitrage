import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, cross
import time


def prep_data(tickers):
    data = pd.read_csv('snp500_daily_prices_volume20-23_reco.csv', index_col='Date', parse_dates=True,
                       date_format='%d/%m/%Y')

    if any(data[data['tic'] == tickers[0]].loc[:, 'Close']
           - data[data['tic'] == tickers[1]].loc[:, 'Close'] < 0):
        data_sprd = data[data['tic'] == tickers[1]].loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']] - \
                    data[data['tic'] == tickers[0]].loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
    else:
        data_sprd = data[data['tic'] == tickers[0]].loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']] - \
                    data[data['tic'] == tickers[1]].loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]

    data_sprd['mean_sprd'] = data_sprd['Close'].rolling(506, center=False).mean()
    data_sprd['stdv_sprd'] = data_sprd['Close'].rolling(506, center=False).std()

    # data_sprd['mean_sprd'] = data_sprd['Close'][:506].mean()
    # data_sprd['stdv_sprd'] = data_sprd['Close'][:506].std()

    data_sprd = data_sprd.iloc[505:, :]

    return data_sprd


params = [
          # (3.0, 4.0, 0.1, 45),  # 'CLF', 'X'
          # (2.5, 3.5, 0.1, 30),  # 'IEX', 'ITW'
          # (1.5, 2.5, 0.1, 55),  # 'CDNS', 'SNPS'
          # (1.5, 2.5, 0.1, 35),  # 'EL', 'ETN'
          # (2.5, 3.0, 0.1, 35),  # 'AME', 'NSC'
          # (2.5, 3.0, 0.1, 50),  # 'SLB', 'SLG'
          # (2.0, 2.5, 0.1, 60),  # 'AN', 'URI'
          # (1.5, 2.0, 0.1, 30),  # 'BXP', 'KMI'
          # (1.5, 2.5, 0.6, 35),  # 'AVB', 'UDR'
          # (2.5, 3.0, 0.1, 30),  # 'ELV', 'UNH'
          # (1.0, 2.0, 0.4, 50),  # 'CMA', 'FITB'
          # (2.5, 3.0, 0.1, 30),  # 'BA', 'UAL'
          # (1.5, 2.0, 0.1, 60),  # 'NCR', 'PVH'
          # (1.5, 2.0, 0.6, 30),  # 'UNM', 'VTR'
          # (1.5, 2.5, 0.1, 30),  # 'FRT', 'REG'
          (3.0, 3.5, 0.1, 30),  # 'L', 'USB'
          ]

for i in range(len(params)):

    class PairsTradingStrategy(Strategy):
        nstdv, stop_loss, size, hold_days = params[i]
        exit = 0.0

        def init(self):
            super().init()

            self.last_trade_date = self.data.index[0]

            def z_score_func(c):
                return (c - self.data['mean_sprd'][-1]) / self.data['stdv_sprd'][-1]

            self.zsi = self.I(z_score_func, self.data.Close, overlay=False)

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


    pairs = [
        # ('CLF', 'X'),
        # ('IEX', 'ITW'),
        # ('CDNS', 'SNPS'),
        # ('EL', 'ETN'),
        # ('AME', 'NSC'),
        # ('SLB', 'SLG'),
        # ('AN', 'URI'),
        # ('BXP', 'KMI'),
        # ('AVB', 'UDR'),
        # ('ELV', 'UNH'),
        # ('CMA', 'FITB'),
        # ('BA', 'UAL'),
        # ('NCR', 'PVH'),
        # ('UNM', 'VTR'),
        # ('FRT', 'REG'),
        ('L', 'USB')
    ]

    aum = 10000000

    dataf = prep_data(pairs[i])

    bt = Backtest(dataf, PairsTradingStrategy, commission=.002, cash=(aum / len(pairs)), trade_on_close=True)

    stats = bt.run()
    print(stats)

    # opt = bt.optimize(nstdv=list(np.arange(1.0, 4.5, 0.5)),
    #                   stop_loss=list(np.arange(2.0, 5.5, 0.5)),
    #                   constraint=lambda x: x.nstdv < x.stop_loss,
    #                   size=list(np.arange(0.05, 1.00, 0.05)),
    #                   hold_days=list(np.arange(30, 60, 5)),
    #                   maximize='Return [%]')
    #
    # opt_params = opt._strategy
    # print(opt_params)
    # print(opt)

    returns_stream = stats._equity_curve
    column = str(pairs[i][0]) + "_" + str(pairs[i][1])
    pnl = pd.DataFrame({column: pd.Series(returns_stream['Equity'].pct_change())}).transpose()
    pnl.to_csv(f"{column}_pnl_oos_rolling.csv")
    stats.to_csv(f"{column}_stats_rolling.csv")

    bt.plot(filename=f"{column}.html")
