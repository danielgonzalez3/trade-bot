import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import date
from jugaad_data.nse import stock_df
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from config import NEIGHBOURS_COUNT, MAX_BARS_BACK

def preprocess_data(df, max_bars_back):
    # Reverse and truncate the DataFrame
    df = df.iloc[::-1]
    df = df.tail(max_bars_back).reset_index(drop=True)

    # Scale the data
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df.iloc[:, 2:-2]), columns=df.columns[2:-2])

    return df

def calculate_technical_indicators(df, rsi_period=14, cci_period=20, rsi_period2=9, adx_period=20):
    ndf = pd.DataFrame()
    ndf['RSI'] = ta.rsi(df['CLOSE'], length=rsi_period)
    ndf['CCI'] = ta.cci(df['HIGH'], df['LOW'], df['CLOSE'], length=cci_period)
    ndf['RSI2'] = ta.rsi(df['CLOSE'], length=rsi_period2)
    return ndf

def train_lorentzian_model(ndf, n_neighbors):
    def get_lorentzian_distance(x1, x2, n_features=ndf.shape[1]):
        distance = 0.0
        for i in range(n_features):
            distance += np.log(1 + abs(x1[i] - x2[i]))
        return distance
    
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=get_lorentzian_distance)
    model.fit(ndf.iloc[50:ndf.shape[0]-1].values)
    return model

def make_predictions(model, ndf, future_candle=10):
    # Use the trained model to make predictions
    # Find the neighbors of the most recent data point
    dis, nbrs = model.kneighbors(np.expand_dims(ndf.iloc[-1], axis=0))
    
    res = []
    time = []
    
    for i in nbrs[0]:
        # Get the prices for the next 'future_candle' periods
        vec = price_df['CLOSE'].iloc[i:i + future_candle]
        t_time = price_df['DATE'].iloc[i:i + future_candle]
        time.append(t_time)
        res.append(vec)
    
    res = np.array(res)
    
    return res, time
if __name__ == "__main__":
    price_df = stock_df(symbol="SBIN", from_date=date(2022, 1, 1), to_date=date(2023, 10, 1), series="EQ")
    df = preprocess_data(price_df, max_bars_back=MAX_BARS_BACK)
    ndf = calculate_technical_indicators(df)
    model = train_lorentzian_model(ndf, n_neighbors=NEIGHBOURS_COUNT)
    
    predictions, time = make_predictions(model, ndf)
    
    plt.plot(time[0], predictions[0], label='Expected', color='blue')
    plt.plot(time[0], price_df['CLOSE'].iloc[-len(predictions[0]):], label='Predicted', color='red')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Expected vs. Actual Results')
    plt.show()
