import requests
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

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

# SL / TP Calculation (ICT style light version)
def calculate_sl_tp(df, signal):
    if signal == "BUY":
        sl = df['low'].iloc[-10:].min()
        tp = df['close'].iloc[-1] + (df['close'].iloc[-1] - sl) * 2
    elif signal == "SELL":
        sl = df['high'].iloc[-10:].max()
        tp = df['close'].iloc[-1] - (sl - df['close'].iloc[-1]) * 2
    else:
        return None, None

    return round(sl, 2), round(tp, 2)

#Multi-Timeframe Confirmation (M5 + H1)
def get_data(interval):
    url = f"https://api.twelvedata.com/time_series?apikey={TWELVE_DATA_API}&symbol=XAU/USD&interval={interval}&outputsize=2500&timezone=Africa/Nairobi"
    data = requests.get(url).json()
    df = pd.DataFrame(data["values"])
    df = df[::-1].reset_index(drop=True)  # reverse order
    df['datetime'] = pd.to_datetime(df['datetime'])
    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    return df

#Predicted Exit Time
def estimate_exit_time(pred_df, tp, signal, y_timestamp):
    for i, price in enumerate(pred_df["close"]):
        if signal == "BUY" and price >= tp:
            return y_timestamp.iloc[i]
        elif signal == "SELL" and price <= tp:
            return y_timestamp.iloc[i]
    return y_timestamp.iloc[-1]

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
kline_df = df.loc[:lookback+pred_len-1]

# visualize
plot_prediction(kline_df, pred_df)

# Confirmation rule
if bos_m5 == "BULLISH_BOS" and bos_h1 == "BULLISH_BOS":
    trend = "BUY"
elif bos_m5 == "BEARISH_BOS" and bos_h1 == "BEARISH_BOS":
    trend = "SELL"
else:
    trend = "NO TRADE"

# Signal Engine
current_price = float(df["close"].iloc[-1])
predicted_price = float(pred_df["close"].iloc[-1])

current_time = df["datetime"].iloc[-1]
predicted_time = y_timestamp.iloc[-1]

bos = detect_bos(df)

if signal == "BUY" and bos == "BULLISH_BOS":
    final_signal = "BUY"
elif signal == "SELL" and bos == "BEARISH_BOS":
    final_signal = "SELL"
else:
    final_signal = "NO TRADE"

sl, tp = calculate_sl_tp(df, final_signal)
exit_time = estimate_exit_time(pred_df, tp, final_signal, y_timestamp)

# Send Signal
msg = f"""
🚀 XAUUSD SIGNAL

Signal: {final_signal}
Entry: {current_price}

SL: {sl}
TP: {tp}

Current Time: {current_time}
Estimated Exit: {exit_time}
"""

send_signal(msg)

chart_path = plot_and_save(kline_df, pred_df)
send_photo(chart_path)
