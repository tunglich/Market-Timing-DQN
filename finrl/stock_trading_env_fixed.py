"""Subclass of FinRL StockTradingEnv that fixes the single-stock initial-position bug.

Upstream `_initiate_state` hard-codes `[0] * stock_dim` for single-stock,
ignoring the `num_stock_shares` argument. This subclass routes the initial
state through the same multi-stock logic so that pre-loaded positions
actually appear in the agent's observation.
"""
from __future__ import annotations

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


class StockTradingEnvFixed(StockTradingEnv):
    def _initiate_state(self):
        if self.initial:
            close_list = (
                self.data.close.values.tolist()
                if hasattr(self.data.close, "values")
                else [float(self.data.close)]
            )
            state = (
                [self.initial_amount]
                + close_list
                + list(self.num_stock_shares)
                + sum(
                    (
                        (self.data[tech].values.tolist()
                         if hasattr(self.data[tech], "values")
                         else [float(self.data[tech])])
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )
            return state
        return super()._initiate_state()
