from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
from functools import reduce
import talib.abstract as ta


class MeanReversionTrendStrategy(IStrategy):
    """
    Mean Reversion inside bullish trend

    الفكرة:
    - ندخل فقط إذا السوق العام صاعد
    - نبحث عن هبوط مبالغ فيه ثم بداية ارتداد
    - لا يوجد AI في قرار الدخول
    """

    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = False

    use_exit_signal = True
    exit_profit_only = False
    process_only_new_candles = True
    startup_candle_count = 250

    # عائد ثابت بسيط
    minimal_roi = {
        "0": 0.06
    }

    # وقف خسارة
    stoploss = -0.03

    # بدون trailing في البداية حتى لا يفسد الاختبار
    trailing_stop = False

    # Parameters قابلة للتعديل لاحقًا بالهايبرأوبت
    buy_rsi_threshold = IntParameter(20, 35, default=30, space="buy")
    sell_rsi_threshold = IntParameter(55, 75, default=60, space="sell")
    volume_multiplier = DecimalParameter(1.0, 2.0, default=1.2, decimals=2, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # EMAs
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        # Volume mean
        dataframe["volume_mean_20"] = dataframe["volume"].rolling(20).mean()

        # شمعة حمراء / خضراء للمساعدة
        dataframe["is_red"] = dataframe["close"] < dataframe["open"]
        dataframe["is_green"] = dataframe["close"] > dataframe["open"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # 1) الاتجاه العام صاعد
        conditions.append(dataframe["close"] > dataframe["ema200"])

        # 2) هبوط ثم بداية ارتداد في RSI
        # الشمعة السابقة تحت 30، والحالية أعلى من السابقة
        conditions.append(dataframe["rsi"].shift(1) < self.buy_rsi_threshold.value)
        conditions.append(dataframe["rsi"] > dataframe["rsi"].shift(1))

        # 3) حجم أعلى من المتوسط
        conditions.append(
            dataframe["volume"] > dataframe["volume_mean_20"] * self.volume_multiplier.value
        )

        # 4) تأكيد بسيط: الشمعة الحالية خضراء
        conditions.append(dataframe["is_green"] == True)

        # 5) نتجنب الدخول لو الحجم صفر
        conditions.append(dataframe["volume"] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                ["enter_long", "enter_tag"]
            ] = (1, "mean_reversion_bull")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # خروج إذا صار RSI مرتفعًا
        conditions.append(dataframe["rsi"] > self.sell_rsi_threshold.value)

        # أو بدأ السعر يضعف تحت EMA50
        conditions.append(dataframe["close"] < dataframe["ema50"])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                ["exit_long", "exit_tag"]
            ] = (1, "rsi_high_and_weak_price")

        return dataframe
