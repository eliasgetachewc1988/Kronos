import requests
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from model import Kronos, KronosTokenizer, KronosPredictor


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


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 3. Prepare Data
url = "https://api.twelvedata.com/time_series?apikey=3617d3ff0ca247aeaa7fcb04d0760b66&symbol=XAU/USD&interval=5min&outputsize=2500"
data = requests.get(url).json()

df = pd.DataFrame(data["values"])
df = df[::-1].reset_index(drop=True)  # reverse order

df['datetime'] = pd.to_datetime(df['datetime'])

lookback = 400
pred_len = 120

x_df = df.iloc[:lookback][['open', 'high', 'low', 'close']]
x_timestamp = df.iloc[:lookback]['datetime']
y_timestamp = df.iloc[lookback:lookback+pred_len]['datetime']

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

# Signal Engine
current_price = df["close"].iloc[-1]
predicted_price = pred_df[-1]

if predicted_price > current_price * 1.002:
    signal = "BUY"
elif predicted_price < current_price * 0.998:
    signal = "SELL"
else:
    signal = "NO TRADE"

# Signal Engine
TOKEN = "5289027180:AAEFDR3KUWSn0MzhWpawQF5RFJZ7ar2fluY"
CHAT_ID = "346632926"

def send_signal(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

# Send Signal
send_signal(f"{signal} XAUUSD\nPrice: {current_price}")
