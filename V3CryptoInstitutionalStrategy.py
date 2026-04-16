# pragma pylint: disable=missing-module-docstring, invalid-name, pointless-string-statement

from __future__ import annotations

from typing import Tuple
import logging

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
    IntParameter,
    merge_informative_pair,
)

logger = logging.getLogger(__name__)


def find_pivot_lows(series: np.ndarray, left: int = 5, right: int = 5) -> np.ndarray:
    pivots = np.full(len(series), np.nan, dtype=float)
    for i in range(left, len(series) - right):
        window = series[i - left: i + right + 1]
        center = series[i]
        if np.isfinite(center) and center == np.nanmin(window):
            pivots[i] = center
    return pivots


def find_pivot_highs(series: np.ndarray, left: int = 5, right: int = 5) -> np.ndarray:
    pivots = np.full(len(series), np.nan, dtype=float)
    for i in range(left, len(series) - right):
        window = series[i - left: i + right + 1]
        center = series[i]
        if np.isfinite(center) and center == np.nanmax(window):
            pivots[i] = center
    return pivots


def detect_hh_hl_structure(
    highs: np.ndarray,
    lows: np.ndarray,
    pivot_highs: np.ndarray,
    pivot_lows: np.ndarray,
) -> Tuple[bool, float, float, float, float]:
    high_idx = [i for i, v in enumerate(pivot_highs) if np.isfinite(v)]
    low_idx  = [i for i, v in enumerate(pivot_lows)  if np.isfinite(v)]

    if len(high_idx) < 2 or len(low_idx) < 2:
        return False, np.nan, np.nan, np.nan, np.nan

    prev_hh_idx, last_hh_idx = high_idx[-2], high_idx[-1]
    prev_hl_idx, last_hl_idx = low_idx[-2],  low_idx[-1]

    prev_hh = highs[prev_hh_idx]
    last_hh = highs[last_hh_idx]
    prev_hl = lows[prev_hl_idx]
    last_hl = lows[last_hl_idx]

    is_bullish = (
        (last_hh > prev_hh)
        and (last_hl > prev_hl)
        and (last_hh_idx > last_hl_idx)
    )

    return is_bullish, float(last_hh), float(last_hl), float(last_hh_idx), float(last_hl_idx)


def get_liquidity_zone(lows: np.ndarray, pivot_lows: np.ndarray, n: int = 3) -> float:
    valid_idx = [i for i, v in enumerate(pivot_lows) if np.isfinite(v)]
    if len(valid_idx) < n:
        return np.nan
    last_n_idx = valid_idx[-n:]
    last_n_lows = [lows[i] for i in last_n_idx]
    return float(np.min(last_n_lows))


class V3CryptoInstitutionalStrategy(IStrategy):
    """
    V3 Crypto Institutional Strategy
    ----------------------------------
    4H trend filter
    1H structure + pullback + liquidity zone
    15m sweep + reclaim + displacement + volume
    """

    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = False

    process_only_new_candles = True
    startup_candle_count = 400

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    stoploss = -0.04
    trailing_stop = False
    use_custom_stoploss = False
    use_custom_exit = True

    minimal_roi = {"0": 100.0}

    sweep_threshold        = DecimalParameter(0.0015, 0.0060, default=0.0030, decimals=4, space="buy")
    volume_multiplier      = DecimalParameter(1.2,   2.2,    default=1.5,    decimals=1, space="buy")
    rr_minimum             = DecimalParameter(1.6,   3.5,    default=2.0,    decimals=1, space="buy")
    atr_sl_buffer          = DecimalParameter(0.15,  0.80,   default=0.35,   decimals=2, space="buy")
    displacement_atr_mult  = DecimalParameter(0.4,   1.2,    default=0.7,    decimals=2, space="buy")
    pivot_left             = IntParameter(3, 7, default=5, space="buy")
    pivot_right            = IntParameter(3, 7, default=5, space="buy")

    def informative_pairs(self):
        if not self.dp:
            return []
        pairs = self.dp.current_whitelist()
        informative = []
        informative += [(pair, "1h") for pair in pairs]
        informative += [(pair, "4h") for pair in pairs]
        return informative

    def _build_informative_1h(self, informative: DataFrame) -> DataFrame:
        inf = informative.copy()
        inf["ema50"]  = ta.EMA(inf, timeperiod=50)
        inf["ema200"] = ta.EMA(inf, timeperiod=200)
        inf["adx"]    = ta.ADX(inf, timeperiod=14)
        inf["atr"]    = ta.ATR(inf, timeperiod=14)

        highs = inf["high"].to_numpy(dtype=float)
        lows  = inf["low"].to_numpy(dtype=float)
        left  = int(self.pivot_left.value)
        right = int(self.pivot_right.value)

        pivot_highs = find_pivot_highs(highs, left=left, right=right)
        pivot_lows  = find_pivot_lows(lows,  left=left, right=right)

        inf["pivot_high"] = pivot_highs
        inf["pivot_low"]  = pivot_lows

        bullish_list, last_hh_list, last_hl_list = [], [], []
        last_hh_idx_list, last_hl_idx_list, liquidity_zone_list = [], [], []

        for i in range(len(inf)):
            ph = pivot_highs[: i + 1]
            pl = pivot_lows[:  i + 1]
            h  = highs[: i + 1]
            l  = lows[:  i + 1]

            bullish, last_hh, last_hl, last_hh_idx, last_hl_idx = detect_hh_hl_structure(h, l, ph, pl)

            bullish_list.append(bullish)
            last_hh_list.append(last_hh)
            last_hl_list.append(last_hl)
            last_hh_idx_list.append(last_hh_idx)
            last_hl_idx_list.append(last_hl_idx)
            liquidity_zone_list.append(get_liquidity_zone(l, pl, n=3))

        inf["is_bullish_structure"] = bullish_list
        inf["last_hh"]              = last_hh_list
        inf["last_hl"]              = last_hl_list
        inf["last_hh_idx"]          = last_hh_idx_list
        inf["last_hl_idx"]          = last_hl_idx_list
        inf["liquidity_zone"]       = liquidity_zone_list

        inf["pullback_valid"] = (
            inf["is_bullish_structure"]
            & (inf["close"] < inf["last_hh"])
            & (inf["close"] > inf["last_hl"])
            & (inf["last_hh_idx"] > inf["last_hl_idx"])
        )

        inf["trend_filter_1h"] = (
            (inf["close"] > inf["ema50"])
            & (inf["ema50"] > inf["ema200"])
            & (inf["adx"] > 18)
        )

        return inf

    def _build_informative_4h(self, informative: DataFrame) -> DataFrame:
        inf = informative.copy()
        inf["ema50"]  = ta.EMA(inf, timeperiod=50)
        inf["ema200"] = ta.EMA(inf, timeperiod=200)
        inf["rsi"]    = ta.RSI(inf, timeperiod=14)

        inf["trend_filter_4h"] = (
            (inf["close"] > inf["ema200"])
            & (inf["ema50"] > inf["ema200"])
            & (inf["rsi"] > 48)
        )

        return inf

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        pair = metadata["pair"]

        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"]   = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"]   = ta.ATR(dataframe, timeperiod=14)

        dataframe["volume_mean_20"] = dataframe["volume"].rolling(20, min_periods=20).mean()

        dataframe["body"]  = (dataframe["close"] - dataframe["open"]).abs()
        dataframe["range"] = (dataframe["high"] - dataframe["low"]).replace(0, np.nan)
        dataframe["close_position"] = (
            (dataframe["close"] - dataframe["low"]) / dataframe["range"]
        ).clip(lower=0, upper=1)

        dataframe["bullish_displacement"] = (
            (dataframe["close"] > dataframe["open"])
            & (dataframe["body"] > dataframe["atr"] * float(self.displacement_atr_mult.value))
            & (dataframe["close"] > dataframe["high"].shift(1))
            & (dataframe["close_position"] > 0.65)
        )

        if not self.dp:
            return dataframe

        informative_1h = self.dp.get_pair_dataframe(pair=pair, timeframe="1h")
        informative_4h = self.dp.get_pair_dataframe(pair=pair, timeframe="4h")

        informative_1h = self._build_informative_1h(informative_1h)
        informative_4h = self._build_informative_4h(informative_4h)

        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, "1h", ffill=True)
        dataframe = merge_informative_pair(dataframe, informative_4h, self.timeframe, "4h", ffill=True)

        dataframe["market_regime_ok"] = (
            dataframe["trend_filter_1h_1h"]
            & dataframe["trend_filter_4h_4h"]
            & dataframe["is_bullish_structure_1h"]
        )

        dataframe["in_pullback"] = (
            dataframe["market_regime_ok"]
            & dataframe["pullback_valid_1h"]
            & (dataframe["close"] < dataframe["last_hh_1h"])
            & (dataframe["close"] > dataframe["last_hl_1h"])
        )

        dataframe["swept_below_zone"] = (
            (dataframe["liquidity_zone_1h"] > 0)
            & (dataframe["low"] < dataframe["liquidity_zone_1h"] * (1.0 - float(self.sweep_threshold.value)))
        )

        dataframe["reclaim_same_candle"] = (
            dataframe["swept_below_zone"]
            & (dataframe["close"] > dataframe["liquidity_zone_1h"])
        )

        dataframe["reclaim_next_candle"] = (
            dataframe["swept_below_zone"].shift(1).fillna(False)
            & (dataframe["close"] > dataframe["liquidity_zone_1h"])
        )

        dataframe["sweep_occurred"] = (
            dataframe["reclaim_same_candle"] | dataframe["reclaim_next_candle"]
        )

        dataframe["sweep_low_price"] = np.where(
            dataframe["reclaim_same_candle"],
            dataframe["low"],
            np.where(dataframe["reclaim_next_candle"], dataframe["low"].shift(1), np.nan),
        )

        dataframe["volume_confirmed"] = (
            dataframe["volume"] > dataframe["volume_mean_20"] * float(self.volume_multiplier.value)
        )

        dataframe["not_overextended"] = (
            dataframe["close"] < dataframe["ema20"] + (dataframe["atr"] * 1.5)
        )

        dataframe["sl_price"] = (
            np.minimum(dataframe["sweep_low_price"], dataframe["liquidity_zone_1h"])
            - (dataframe["atr"] * float(self.atr_sl_buffer.value))
        )

        dataframe["tp_price_structure"] = dataframe["last_hh_1h"]
        dataframe["tp_price_atr_cap"]   = dataframe["close"] + (dataframe["atr"] * 4.0)
        dataframe["tp_price"] = np.minimum(
            dataframe["tp_price_structure"],
            dataframe["tp_price_atr_cap"],
        )

        dataframe["entry_price"] = dataframe["close"]
        dataframe["risk"]        = dataframe["entry_price"] - dataframe["sl_price"]
        dataframe["reward"]      = dataframe["tp_price"]    - dataframe["entry_price"]

        dataframe["rr_ratio"] = np.where(
            (dataframe["risk"] > 0) & (dataframe["reward"] > 0),
            dataframe["reward"] / dataframe["risk"],
            0.0,
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = (
            dataframe["market_regime_ok"]
            & dataframe["in_pullback"]
            & dataframe["sweep_occurred"]
            & dataframe["bullish_displacement"]
            & dataframe["volume_confirmed"]
            & dataframe["not_overextended"]
            & (dataframe["close"] > dataframe["last_hl_1h"])
            & (dataframe["rr_ratio"] >= float(self.rr_minimum.value))
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[conditions, ["enter_long", "enter_tag"]] = (
            1, "v3_sweep_displacement_long"
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"]  = None
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs
    ):
        if not self.dp:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or len(dataframe) < 2:
            return None

        last = dataframe.iloc[-1]
        prev = dataframe.iloc[-2]

        tp_price = last.get("tp_price", np.nan)
        last_hl  = last.get("last_hl_1h", np.nan)
        ema20    = last.get("ema20", np.nan)
        atr      = last.get("atr",  np.nan)
        rsi      = last.get("rsi",  np.nan)
        vol_ok   = bool(last.get("volume_confirmed", False))

        # 1) هدف هيكلي
        if pd.notna(tp_price) and current_rate >= tp_price:
            return "tp_structure"

        # 2) كسر هيكل 1H
        if pd.notna(last_hl) and current_rate < last_hl:
            return "structure_broken"

        # 3) حماية الربح عند ضعف الزخم
        if (
            current_profit > 0.015
            and pd.notna(ema20)
            and current_rate < ema20
            and pd.notna(rsi)
            and rsi < 48
        ):
            return "profit_protect_momentum_loss"

        # 4) تشبع شرائي واضح
        if current_profit > 0.02 and pd.notna(rsi) and rsi > 73:
            return "take_profit_rsi"

        # 5) فشل المتابعة بعد displacement
        if (
            current_profit > 0.008
            and bool(prev.get("bullish_displacement", False))
            and not vol_ok
            and pd.notna(ema20)
            and current_rate < ema20
        ):
            return "follow_through_failed"

        # 6) time stop — صفقة بطيئة
        age_minutes = (current_time - trade.open_date_utc).total_seconds() / 60.0
        if age_minutes > 720 and current_profit < 0.003:
            return "time_stop"

        # 7) رفض سعري قرب القمة
        if (
            current_profit > 0.015
            and pd.notna(atr)
            and (last["high"] - last["close"]) > atr * 0.8
            and pd.notna(rsi)
            and rsi > 65
        ):
            return "rejection_near_high"

        return None
