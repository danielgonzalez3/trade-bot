import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import date
from jugaad_data.nse import stock_df  # Used to fetch stock data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from config import NEIGHBOURS_COUNT, MAX_BARS_BACK

def preprocess_data(og_df, max_bars_back):
    """
    Preprocesses the data by reversing, truncating, and scaling.

    Parameters:
    processed_df (DataFrame): The original data frame.
    max_bars_back (int): The number of recent bars to consider.

    Returns:
    DataFrame: The processed data frame.
    """
    # Keep a copy of the original data for later use
    processed_df = og_df.copy()
    
    # Truncate to keep only the last max_bars_back rows
    #processed_df = processed_df.tail(max_bars_back).reset_index(drop=True)

    # Determine which columns to scale (typically numerical columns)
    # For instance, if you need 'open', 'high', 'low', 'close', and 'volume':
    columns_to_scale = ['open', 'high', 'low', 'close', 'volume', 'confidence']
    
    # Scale the specified columns using MinMaxScaler
    scaler = MinMaxScaler()
    processed_df[columns_to_scale] = scaler.fit_transform(processed_df[columns_to_scale])

    return processed_df

def calculate_technical_indicators(processed_df, rsi_period=14, cci_period=20, rsi_period2=9, adx_period=20):
    """
    Calculates various technical indicators.

    Parameters:
    processed_df (DataFrame): The data frame with stock prices.
    rsi_period (int): Period for RSI calculation.
    cci_period (int): Period for CCI calculation.
    rsi_period2 (int): Another period for RSI calculation.

    Returns:
    DataFrame: A data frame with calculated technical indicators.
    """

    indicator_df = pd.DataFrame()
    indicator_df['RSI'] = ta.rsi(processed_df['close'], length=rsi_period)
    indicator_df['CCI'] = ta.cci(processed_df['high'], processed_df['low'], processed_df['close'], length=cci_period)
    indicator_df['RSI2'] = ta.rsi(processed_df['close'], length=rsi_period2)
    indicator_df['ADX'] = ta.adx(processed_df['high'], processed_df['low'], processed_df['close'], length=adx_period)['ADX_20']
    return indicator_df

def train_lorentzian_model(indicator_df, n_neighbors):
    """
    Trains a model using the Lorentzian distance metric.

    Parameters:
    indicator_df (DataFrame): The data frame with technical indicators.
    n_neighbors (int): The number of neighbors to consider in the model.

    Returns:
    NearestNeighbors: The trained Nearest Neighbors model.
    """
    def get_lorentzian_distance(x1, x2, n_features=indicator_df.shape[1]):
        # Lorentzian distance function
        distance = 0.0
        for i in range(n_features):
            distance += np.log(1 + abs(x1[i] - x2[i]))
        return distance
    
    # Check if indicator_df has enough rows
    if indicator_df.shape[0] > 50:
        training_data = indicator_df.iloc[50:indicator_df.shape[0]-1].values
    else:
        # If not enough rows, use all available data
        training_data = indicator_df.values

    # Initialize and train the Nearest Neighbors model
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=get_lorentzian_distance)
    model.fit(training_data)
    return model

def make_predictions(model, indicator_df, processed_df, future_candle=5):
    """
    Makes predictions using the trained model.

    Parameters:
    model (NearestNeighbors): The trained model.
    indicator_df (DataFrame): The data frame with technical indicators.
    processed_df (DataFrame): The preprocessed data frame.
    future_candle (int): The number of future periods to predict.

    Returns:
    Tuple: A tuple containing the predicted prices and their corresponding times.
    """
    # Use the trained model to find the k-nearest neighbors of the most recent data point in indicator_df
    dis, nbrs = model.kneighbors(np.expand_dims(indicator_df.iloc[-1], axis=0))
    #print("nbrs:\n", nbrs)
    #print("dis:\n", dis)

    # Initialize arrays to store the results
    res = np.empty((NEIGHBOURS_COUNT, future_candle))
    time = np.empty((NEIGHBOURS_COUNT, future_candle))

    # Loop over the indices of the nearest neighbors
    for idx, neighbor_index in enumerate(nbrs[0]):
        # Extract the 'close' prices for the next 'future_candle' periods from processed_df
        res[idx] = processed_df['close'].iloc[neighbor_index:neighbor_index + future_candle].to_numpy()
        # Extract the corresponding 'dateTime' values
        time[idx] = processed_df['dateTime'].iloc[neighbor_index:neighbor_index + future_candle].to_numpy()

    # Convert the timestamp data to pandas Timestamp objects
    time = np.array([[pd.to_datetime(ts, unit='ns') for ts in row] for row in time])
    #print("time:\n", time)
    #print("res:\n", res)

    return res, time

# The trend_direction function calculates the overall trend direction based on increases and decreases
def trend_direction(numbers):
    increases = 0
    decreases = 0

    # Loop through the numbers and count how many times they increase or decrease
    for i in range(1, len(numbers)):
        if numbers[i] > numbers[i-1]:
            increases += 1
        elif numbers[i] < numbers[i-1]:
            decreases += 1

    # Determine the overall trend based on the counts
    if increases > decreases:
        return 1  # Indicates an uptrend
    elif decreases > increases:
        return -1  # Indicates a downtrend
    else:
        return 0  # Indicates no clear trend
    
# The trend_direction_moving_average function calculates the trend direction OF THE PREDICTIONS based on a moving average
def trend_direction_moving_average(numbers, window_size=3):
    if len(numbers) < window_size:
        return "Data insufficient for moving average"

    # Calculate the moving average and then determine the trend direction
    moving_averages = [sum(numbers[i:i+window_size])/window_size for i in range(len(numbers) - window_size + 1)]
    return trend_direction(moving_averages)
    
if __name__ == "__main__":
    # Load data from JSON file instead of fetching from NSE
    #json_filename = 'BTC_USD-ONE_MINUTE-2015_8_1-to-2024_1_31.json'
    #json_filename = 'BTC_USD-FIVE_MINUTE-2015_8_1-to-2024_1_31.json'
    #json_filename = 'BTC_USD-FIFTEEN_MINUTE-2015_8_1-to-2024_1_31.json'
    #json_filename = 'BTC_USD-ONE_HOUR-2015_8_1-to-2024_1_31.json'
    #json_filename = 'BTC_USD-SIX_HOUR-2015_8_1-to-2024_1_31.json'
    json_filename = 'BTC_USD-ONE_DAY-2015_8_1-to-2024_1_31.json'
    #json_filename = 'test.json'
    og_df = pd.read_json(json_filename)

    # Convert columns to appropriate types
    og_df['start'] = pd.to_numeric(og_df['start'])
    og_df['dateTime'] = pd.to_datetime(og_df['dateTime'])
    og_df['high'] = pd.to_numeric(og_df['high'])
    og_df['low'] = pd.to_numeric(og_df['low'])
    og_df['open'] = pd.to_numeric(og_df['open'])
    og_df['close'] = pd.to_numeric(og_df['close'])
    og_df['volume'] = pd.to_numeric(og_df['volume'])
    og_df['confidence'] = 0

    # Reverse the DataFrame to have the most recent data at the end
    og_df = og_df.iloc[::-1]
    og_df = og_df.reset_index(drop=True)

    # Preprocess the data (scaling and other transformations)
    processed_df = preprocess_data(og_df, MAX_BARS_BACK)
    #print("Preprocessed DataFrame:\n", processed_df)
    
    # Calculate technical indicators
    indicator_df = calculate_technical_indicators(processed_df)

    # Handle NaN values, e.g., fill with the first non-NaN value or another strategy (currently using backward filling)
    indicator_df.fillna(method='bfill', inplace=True)
    #print("Final Technical Indicators DataFrame:\n", indicator_df)
    
    confidence = 0
    future_candle = 5

    correct_prediction = 0
    incorrect_prediction = 0
    no_prediction = -1
    prediction_start = 1000

    for curr in range(prediction_start, len(indicator_df)):
        curr_processed_df = processed_df.iloc[:curr]
        curr_indicator_df = indicator_df.iloc[:curr]

        confidence = 0
        # Train the Lorentzian model on the indicator data
        model = train_lorentzian_model(curr_indicator_df, NEIGHBOURS_COUNT)

        # Make predictions using the trained model
        neighbors, time = make_predictions(model, curr_indicator_df, curr_processed_df, future_candle)

        # Calculate a confidence level for the predictions
        for i in range(NEIGHBOURS_COUNT):
            confidence += trend_direction_moving_average(neighbors[i])

        #print("confidence: ", confidence)
        og_df.at[curr, 'confidence'] = confidence

    for curr in range(prediction_start + future_candle, len(og_df) - future_candle):
        # Grab the 'confidence' at current index in og_df
        current_confidence = og_df.at[curr, 'confidence']

        # Grab the next future_candle 'close' values starting at current index in og_df
        next_closes = og_df['close'].iloc[curr:curr + future_candle].to_numpy()
        
        # Get the trend direction of the next_closes
        trend = trend_direction_moving_average(next_closes)

        # Check if our confidence matches future trend
        if current_confidence > 0:
            # trend is up and confidence is up = good
            if trend >= 0:
                correct_prediction += 1
            else:
                incorrect_prediction += 1
        elif current_confidence < 0:
            # trend is up and confidence is down = bad
            if trend > 0:
                incorrect_prediction += 1
            else:
                correct_prediction += 1
        else:
            if trend == 0:
                correct_prediction += 1
            else:  
                no_prediction += 1

    print("correct_prediction: ", correct_prediction)
    print("incorrect_prediction: ", incorrect_prediction)
    print("no_prediction: ", no_prediction)

    processed_df = preprocess_data(og_df, MAX_BARS_BACK)

    # Plotting
    # setting up multplie 3 plots
    figure, axis = plt.subplots(2, 1)

    # plotting real price 
    axis[0].set_title('Price Close Points')
    axis[0].plot(og_df['dateTime'], og_df['close'], label='price', color='blue')

    # plotting model projected 
    #axis[1].set_title('model projected')
    #axis[1].plot(range(future_candle), neighbors[i], label='Predicted', color='red')

    # plotting processed data 
    axis[1].set_title('processed data with confidence')
    axis[1].plot(processed_df['dateTime'], processed_df['close'], label='price', color='red')
    axis[1].plot(processed_df['dateTime'], processed_df['confidence'], label='confidence', color='blue')


    #plt.legend()
    #plt.show()
