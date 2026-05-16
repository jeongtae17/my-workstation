import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

def calculate_indicators(df):
    close = df["Close"]

    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()

    df["EMA20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["EMA50"] = EMAIndicator(close=close, window=50).ema_indicator()

    bb = BollingerBands(close)
    df["BB_HIGH"] = bb.bollinger_hband()
    df["BB_LOW"] = bb.bollinger_lband()
    df["BB_MID"] = bb.bollinger_mavg()

    df["RSI"] = RSIIndicator(close=close).rsi()

    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_HIST"] = macd.macd_diff()

    return df

def generate_signal(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # -----------------------------------------------------------------
    # 출처 1 적용: 시장 국면(Regime) 판단
    # -----------------------------------------------------------------
    is_strong_trend = (latest["MA5"] > latest["MA20"] > latest["MA60"]) and (latest["MACD"] > 0)
    
    # 1. 추세 점수 (Trend Score - 만점 40점)
    trend_score = 0
    if latest["MA5"] > latest["MA20"]: trend_score += 15
    if latest["MA20"] > latest["MA60"]: trend_score += 10
    if latest["Close"] > latest["MA20"]: trend_score += 15

    # 2. 모멘텀 점수 (Momentum Score - 만점 30점)
    momentum_score = 0
    if latest["MACD"] > latest["MACD_SIGNAL"]:
        momentum_score += 15
        if latest["MACD_HIST"] > prev["MACD_HIST"]:  # 히스토그램 확장(에너지 강화)
            momentum_score += 5
            
    # 출처 1&2 교차검증: 강한 추세장일 때는 높은 RSI가 강세 신호임
    if is_strong_trend:
        if latest["RSI"] >= 50: momentum_score += 10  # 추세 지속
    else:
        if 40 <= latest["RSI"] <= 65: momentum_score += 10  # 횡보장 안정적 진입

    # 3. 변동성 및 가격 행동 점수 (Volatility Score - 만점 30점)
    volatility_score = 0
    # 출처 2(존 볼린저) 적용: 밴드 상단 돌파는 강력한 추세 시작 시그널
    if latest["Close"] > latest["BB_HIGH"]:
        volatility_score += 30 if is_strong_trend else 15
    elif latest["Close"] > latest["BB_MID"]:
        volatility_score += 15

    # 국면별 최종 점수 재합산 (가중치 튜닝)
    if is_strong_trend:
        # 추세와 모멘텀에 80% 비중 분배
        score = int((trend_score * 1.1) + (momentum_score * 1.1) + (volatility_score * 0.6))
    else:
        score = int(trend_score + momentum_score + volatility_score)
        
    score = min(100, max(0, score))

    # 시그널 판단
    if score >= 80: signal = "STRONG BUY"
    elif score >= 60: signal = "BUY"
    elif score >= 40: signal = "HOLD"
    else: signal = "SELL"

    return signal, score

def calculate_probability(score, rsi, macd, df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    avg_volume = df["Volume"].tail(20).mean()

    # 1. 기술적 점수 기반 확률 (최대 45%)
    base_prob = (score / 100) * 45

    # 2. 출처 3(리차드 암스) 적용: 거래량 및 캔들 정밀 분석 (최대 25%)
    volume_factor = 0
    is_bullish_candle = latest["Close"] > latest["Open"]
    
    if latest["Volume"] > avg_volume * 1.5:
        volume_factor = 25 if is_bullish_candle else -20  # 거래량 실린 대음봉은 확률 급감
    elif latest["Volume"] > avg_volume:
        volume_factor = 15 if is_bullish_candle else -10
    elif latest["Volume"] < avg_volume * 0.7:
         volume_factor = -5  # 거래량 없는 상승은 가짜 돌파 확률 증가

    # 3. 위치 및 과열 보정 (최대 20%)
    pos_factor = 10
    # 볼린저 밴드 하단 지지 반등 현상
    if prev["Low"] <= prev["BB_LOW"] and latest["Close"] > prev["Close"]:
        pos_factor += 10
        
    # 출처 1, 2 기반 RSI 보정 고도화
    is_strong_trend = (latest["MA5"] > latest["MA20"] > latest["MA60"])
    if rsi > 78:
        # 강력 정배열 상승장이 아니라면 과열로 판단하여 감점
        if not is_strong_trend:
            pos_factor -= 15
    elif rsi < 25:
        pos_factor += 5  # 과매도 반등 기대

    # 4. 추세 지속성 (최대 10%)
    trend_factor = 0
    if macd > 0 and latest["MACD_HIST"] > prev["MACD_HIST"]:
        trend_factor = 10

    # 최종 확률 합산 (기본 확률 25%에서 시작하여 합산)
    final_prob = 25 + base_prob + volume_factor + pos_factor + trend_factor
    
    # 현실적인 기관 투자 한계 확률 적용 (완벽한 100% 예측은 불가능하므로 95% 캡)
    final_prob = max(5, min(95, int(final_prob)))
    
    return final_prob

def determine_style(info, df, rsi):
    quote_type = info.get("quoteType", "")
    name = info.get("name", "").lower()
    latest = df.iloc[-1]

    ma20 = latest.get("MA20", latest["Close"])
    ma60 = latest.get("MA60", latest["Close"])

    if quote_type == "ETF":
        return "단기" if rsi >= 75 else "장기"

    mega_caps = ["apple", "microsoft", "nvidia", "amazon", "meta", "alphabet", "tesla"]
    if any(x in name for x in mega_caps):
        if ma20 > ma60:
            return "장기"

    if rsi >= 70 and (latest["Close"] < latest["MA5"]): # 과열 상태에서 단기 꺾임 발생 시
        return "단기"

    return "중장기"