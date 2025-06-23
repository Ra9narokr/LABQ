import numpy as np
from scipy.stats import norm


class Option:
    def __init__(self, opt_type, underlying, strike, vol, maturity,position, multiplier=1.0):
        self.opt_type = opt_type        # str: 'call' or 'put' 看涨/看跌
        self.underlying = underlying    # int: option index 期权索引
        self.strike = strike            # float: strike price 期权执行价
        self.vol = vol                  # float: volatility 波动率
        self.maturity = maturity        # float: expiry date(yr) 期权到期时间（年）
        self.position = position        # int: + long, – short 仓位（+ 多，- 空）
        self.multiplier = multiplier    # float: contract multiplier 合约乘数

    def analytic_optdelta(self, S0, r):
        T = self.maturity
        d1 = (np.log(S0 / self.strike) + (r + 0.5 * self.vol ** 2) * T) / (self.vol * np.sqrt(T))
        if self.opt_type.lower() == "call":
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1.0

class OptionPortfolio:
    def __init__(self, r, price_min, price_max, options):
        self.r = r #    risk-free interest rate 无风险利率
        self.price_min = price_min
        self.price_max = price_max
        self.options = options

    def analytic_portdelta(self, S0_vec):
        total = 0.0
        for opt in self.options:
            S0_i = S0_vec[opt.underlying]
            delta_i = opt.analytic_optdelta(S0_i, self.r)
            total += opt.position * opt.multiplier * delta_i
        return total

    def alter(self, price):
        return (price - self.price_min)/(self.price_max - self.price_min)

    def to_initial(self, price):
        return self.price_min + price * (self.price_max - self.price_min)

    def target_function(self, S):
        if np.isscalar(S):
            S = [S]
        S_initial = [self.to_initial(i) for i in S]
        return self.analytic_portdelta(S_initial)