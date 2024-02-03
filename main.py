import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import date
from jugaad_data.nse import stock_df  # Used to fetch stock data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from config import NEIGHBOURS_COUNT, MAX_BARS_BACK

def preprocess_data(df, max_bars_back):
    """
    Preprocesses the data by reversing, truncating, and scaling.

    Parameters:
    df (DataFrame): The original data frame.
    max_bars_back (int): The number of recent bars to consider.

    Returns:
    DataFrame: The processed data frame.
    """
    # Reverse the DataFrame to have the most recent data at the end
    df = df.iloc[::-1]
    # Truncate to keep only the last max_bars_back rows
    df = df.tail(max_bars_back).reset_index(drop=True)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df.iloc[:, 2:-2]), columns=df.columns[2:-2])

    return df

def calculate_technical_indicators(df, rsi_period=14, cci_period=20, rsi_period2=9, adx_period=20):
    """
    Calculates various technical indicators.

    Parameters:
    df (DataFrame): The data frame with stock prices.
    rsi_period (int): Period for RSI calculation.
    cci_period (int): Period for CCI calculation.
    rsi_period2 (int): Another period for RSI calculation.

    Returns:
    DataFrame: A data frame with calculated technical indicators.
    """
    ndf = pd.DataFrame()
    ndf['RSI'] = ta.rsi(df['CLOSE'], length=rsi_period)
    ndf['CCI'] = ta.cci(df['HIGH'], df['LOW'], df['CLOSE'], length=cci_period)
    ndf['RSI2'] = ta.rsi(df['CLOSE'], length=rsi_period2)
    return ndf

def train_lorentzian_model(ndf, n_neighbors):
    """
    Trains a model using the Lorentzian distance metric.

    Parameters:
    ndf (DataFrame): The data frame with technical indicators.
    n_neighbors (int): The number of neighbors to consider in the model.

    Returns:
    NearestNeighbors: The trained Nearest Neighbors model.
    """
    def get_lorentzian_distance(x1, x2, n_features=ndf.shape[1]):
        # Lorentzian distance function
        distance = 0.0
        for i in range(n_features):
            distance += np.log(1 + abs(x1[i] - x2[i]))
        return distance
    
    # Initialize and train the Nearest Neighbors model
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=get_lorentzian_distance)
    model.fit(ndf.iloc[50:ndf.shape[0]-1].values)
    return model

def make_predictions(model, ndf, future_candle=10):
    """
    Makes predictions using the trained model.

    Parameters:
    model (NearestNeighbors): The trained model.
    ndf (DataFrame): The data frame with technical indicators.
    future_candle (int): The number of future periods to predict.

    Returns:
    Tuple: A tuple containing the predicted prices and their corresponding times.
    """
    # Use the trained model to make predictions
    # Find the neighbors of the most recent data point
    dis, nbrs = model.kneighbors(np.expand_dims(ndf.iloc[-1], axis=0))
    
    res = []
    time = []
    
    for i in nbrs[0]:
        # Get the prices for the next 'future_candle' periods
        vec = price_df['close'].iloc[i:i + future_candle]
        t_time = price_df['start'].iloc[i:i + future_candle]
        time.append(t_time)
        res.append(vec)
    
    res = np.array(res)
    
    return res, time

if __name__ == "__main__":
    # Load data from JSON file instead of fetching from NSE
    json_filename = 'BTC_USD-FIFTEEN_MINUTE-2015_8_1-to-2024_1_31.json'
    price_df = pd.read_json(json_filename)

    # Convert columns to appropriate types
    price_df['start'] = pd.to_datetime(price_df['start'])
    price_df['dateTime'] = pd.to_datetime(price_df['dateTime'])
    price_df['high'] = pd.to_numeric(price_df['high'])
    price_df['low'] = pd.to_numeric(price_df['low'])
    price_df['open'] = pd.to_numeric(price_df['open'])
    price_df['close'] = pd.to_numeric(price_df['close'])
    price_df['volume'] = pd.to_numeric(price_df['volume'])

    # Preprocess the data
    df = preprocess_data(price_df, MAX_BARS_BACK)
    
    # Calculate technical indicators
    ndf = calculate_technical_indicators(df)
    
    # Train the Lorentzian model
    model = train_lorentzian_model(ndf, NEIGHBOURS_COUNT)
    
    # Make predictions
    predictions, time = make_predictions(model, ndf)
    
    # Plotting
    plt.plot(time[0], predictions[0], label='Expected', color='blue')
    plt.plot(time[0], price_df['close'].iloc[-len(predictions[0]):], label='Predicted', color='red')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Expected vs. Actual Results')
    plt.show()
