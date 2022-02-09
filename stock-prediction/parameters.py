import os
import time
from tensorflow.python.keras.layers import LSTM

# Window size or the sequence length
N_STEPS = 30
# Lookup step, 1 is the next day
LOOKUP_STEP = 1
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
#FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'adjclose', 'volume',
#                   'Dayofweek', 'Is_quarter_end', 'Is_quarter_start',
#                    'Year', ]
# date now
date_now = time.strftime("%Y-%m-%d")
#date_now = "2020-08-04"
### model parameters

N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = True

### training parameters

# mean squared error loss mse mae hubber_loss
LOSS = "mse"
# rmsprop adam
OPTIMIZER = "adam"
BATCH_SIZE = 256
EPOCHS = 300

# Apple stock market
ticker = "AMZN"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save
model_name = f"{date_now}_{ticker}-{LOSS}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"

if BIDIRECTIONAL:
    model_name += "-b"
