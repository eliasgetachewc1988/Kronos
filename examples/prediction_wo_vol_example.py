import requests
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from model import Kronos, KronosTokenizer, KronosPredictor

TOKEN = "5289027180:AAEFDR3KUWSn0MzhWpawQF5RFJZ7ar2fluY"
CHAT_ID = "346632926"
TWELVE_DATA_API = "3617d3ff0ca247aeaa7fcb04d0760b66"

def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax.set_ylabel('Close Price', fontsize=14)
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# Break of Structure
def detect_bos(df):
    highs = df['high']
    lows = df['low']

    prev_high = highs.iloc[-20:-1].max()
    prev_low = lows.iloc[-20:-1].min()

    current_close = df['close'].iloc[-1]

    if current_close > prev_high:
        return "BULLISH_BOS"
    elif current_close < prev_low:
        return "BEARISH_BOS"
    else:
        return "NO_BOS"

# SL / TP Calculation From Prediction
def calculate_sl_tp_from_prediction(pred_df, current_price, signal):
    pred_prices = pred_df["close"].astype(float).values

    if signal == "BUY":
        tp = max(pred_prices)

        # SL = lowest dip before TP is reached
        tp_index = pred_prices.argmax()
        sl = min(pred_prices[:tp_index+1]) if tp_index > 0 else current_price * 0.998

    elif signal == "SELL":
        tp = min(pred_prices)

        tp_index = pred_prices.argmin()
        sl = max(pred_prices[:tp_index+1]) if tp_index > 0 else current_price * 1.002

    else:
        return None, None

    return round(sl, 2), round(tp, 2)

#Multi-Timeframe Confirmation (M5 + H1)
def get_data(interval):
    url = f"https://api.twelvedata.com/time_series?apikey={TWELVE_DATA_API}&symbol=XAU/USD&interval={interval}&outputsize=2500&timezone=Africa/Nairobi"
    data = requests.get(url).json()
    df = pd.DataFrame(data["values"])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    return df

#Predicted Exit Time
def build_future_timestamps(last_time, steps, interval_minutes=5):
    return [last_time + timedelta(minutes=interval_minutes * (i+1)) for i in range(steps)]

#Predicted Exit Time
def estimate_exit_time(pred_df, tp, signal, future_timestamps):
    pred_prices = pred_df["close"].astype(float).values

    for i, price in enumerate(pred_prices):
        if signal == "BUY" and price >= tp:
            return future_timestamps[i]
        elif signal == "SELL" and price <= tp:
            return future_timestamps[i]

    return future_timestamps[-1]

#Modify Plot
def plot_and_save(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]

    plt.figure(figsize=(10,5))
    plt.plot(kline_df['close'], label='Actual')
    plt.plot(pred_df['close'], label='Prediction')
    plt.legend()
    plt.grid()

    file_path = "chart.png"
    plt.savefig(file_path)
    plt.close()

    return file_path

def plot_prediction2(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax.set_ylabel('Close Price', fontsize=14)
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True)

    plt.tight_layout()

    file_path = "chart.png"
    plt.savefig(file_path)
    plt.close()

    return file_path

#Confidence Engine
def calculate_confidence(current_price, predicted_price, bos_m5, bos_h1, sl, tp, signal):
    score = 0
    max_score = 100

    # 1. Prediction Strength (0–40)
    move_pct = abs(predicted_price - current_price) / current_price

    if move_pct > 0.003:  # strong move
        score += 40
    elif move_pct > 0.002:
        score += 30
    elif move_pct > 0.001:
        score += 20
    else:
        score += 10

    # 2. BOS Alignment (0–25)
    if signal == "BUY" and bos_m5 == "BULLISH_BOS":
        score += 12
    if signal == "SELL" and bos_m5 == "BEARISH_BOS":
        score += 12

    if signal == "BUY" and bos_h1 == "BULLISH_BOS":
        score += 13
    if signal == "SELL" and bos_h1 == "BEARISH_BOS":
        score += 13

    # 3. Risk Reward Quality (0–20)
    if sl is not None and tp is not None:
        risk = abs(current_price - sl)
        reward = abs(tp - current_price)

        if risk > 0:
            rr = reward / risk
            if rr >= 2:
                score += 20
            elif rr >= 1.5:
                score += 15
            elif rr >= 1:
                score += 10

    # 4. Signal Agreement (0–15)
    if signal != "NO TRADE":
        score += 15

    confidence = min(round(score, 2), 100)
    return confidence

#Confidence Tiers
def confidence_label(conf):
    if conf >= 80:
        return "🔥 HIGH PROBABILITY"
    elif conf >= 65:
        return "⚡ MEDIUM"
    elif conf >= 50:
        return "⚠️ LOW"
    else:
        return "❌ AVOID"

#Send message to Telegram
def send_signal(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

#Send Chart Image to Telegram
def send_photo(file_path):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(file_path, "rb") as photo:
        requests.post(url, data={"chat_id": CHAT_ID}, files={"photo": photo})

# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 3. Prepare Data
df_m5 = get_data("5min")
df_h1 = get_data("1h")

bos_m5 = detect_bos(df_m5)
bos_h1 = detect_bos(df_h1)

lookback = 400
pred_len = 120

last_time = df_m5["datetime"].iloc[-1]
yf_timestamp = build_future_timestamps(last_time, pred_len, interval_minutes=5)

x_df = df_m5.iloc[:lookback][['open', 'high', 'low', 'close']]
x_timestamp = df_m5.iloc[:lookback]['datetime']
y_timestamp = df_m5.iloc[lookback:lookback+pred_len]['datetime']

# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
kline_df = df_m5.loc[:lookback+pred_len-1]

# visualize
plot_prediction(kline_df, pred_df)

# Signal Engine
current_price = float(df_m5["close"].iloc[-1])
predicted_price = float(pred_df["close"].iloc[-1])
current_time = df_m5["datetime"].iloc[-1]

# Confirmation rule
if bos_m5 == "BULLISH_BOS" and bos_h1 == "BULLISH_BOS":
    trend = "BUY"
elif bos_m5 == "BEARISH_BOS" and bos_h1 == "BEARISH_BOS":
    trend = "SELL"
else:
    trend = "NO TRADE"

bos = detect_bos(df_m5)

if trend == "BUY" and bos == "BULLISH_BOS":
    final_signal = "BUY"
elif trend == "SELL" and bos == "BEARISH_BOS":
    final_signal = "SELL"
else:
    final_signal = "NO TRADE"

sl, tp = calculate_sl_tp_from_prediction(pred_df, current_price, final_signal)

exit_time = estimate_exit_time(
    pred_df,
    tp,
    final_signal,
    yf_timestamp
)

if final_signal == "BUY" and tp <= current_price:
    final_signal = "NO TRADE"

if final_signal == "SELL" and tp >= current_price:
    final_signal = "NO TRADE"

# Confidence Signal
confidence = calculate_confidence(
    current_price,
    predicted_price,
    bos_m5,
    bos_h1,
    sl,
    tp,
    final_signal
)

if confidence < 60:
    final_signal = "NO TRADE"
    
label = confidence_label(confidence)

# Send Signal
msg = f"""
🚀 XAUUSD SIGNAL

Signal: {final_signal}
Confidence: {confidence}% ({label})

Current Price/Entry: {current_price}
Predicted Price: {predicted_price}

SL: {sl}
TP: {tp}

5m BOS: {bos_m5}
1h BOS: {bos_h1}

Current Time: {current_time}
Estimated Exit: {exit_time}
"""

send_signal(msg)

chart_path = plot_prediction2(kline_df, pred_df)
send_photo(chart_path)
