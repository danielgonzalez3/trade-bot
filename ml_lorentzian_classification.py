import config
import sys
from MLExtensions import MLExtensions as ml
import kernelFunctions as kernels
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import date
from jugaad_data.nse import stock_df  # Used to fetch stock data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def series_from(feature_string, close, high, low, f_paramA, f_paramB):
    match feature_string:
        case 'RSI' | 'RSI2':
            return ml.n_rsi(close, f_paramA, f_paramB)
        case 'WT':
            return ml.n_wt((close + high + low) / 3, f_paramA, f_paramB)
        case 'CCI':
            return ml.n_cci(high, low, close, f_paramA, f_paramB)
        case 'ADX':
            return ml.n_adx(high, low, close, f_paramA)

def preprocess_data(og_df, MAX_BARS_BACK):
    """
    Preprocesses the data by reversing, truncating, and scaling.

    Parameters:
    processed_df (DataFrame): The original data frame.
    config.MAX_BARS_BACK (int): The number of recent bars to consider.

    Returns:
    DataFrame: The processed data frame.
    """
    # Keep a copy of the original data for later use
    processed_df = og_df.copy()
    
    # Truncate to keep only the last config.MAX_BARS_BACK rows
    #processed_df = processed_df.tail(config.MAX_BARS_BACK).reset_index(drop=True)

    # Determine which columns to scale (typically numerical columns)
    # For instance, if you need 'open', 'high', 'low', 'close', and 'volume':
    columns_to_scale = ['open', 'high', 'low', 'close', 'volume', 'confidence', 'false_up', 'false_down', 'true_up', 'true_down', 'no_prediction']
    
    # Scale the specified columns using MinMaxScaler
    scaler = MinMaxScaler()
    processed_df[columns_to_scale] = scaler.fit_transform(processed_df[columns_to_scale])

    return processed_df

def calculate_technical_indicators(data_frame):
    """
    Calculates various technical indicators.

    Parameters:
    data_frame (DataFrame): The data frame with stock prices.

    Returns:
    DataFrame: A data frame with calculated technical indicators.
    """

    indicator_df = pd.DataFrame()
    indicator_df['f1'] = series_from(config.f1_string, data_frame['close'], data_frame['high'], data_frame['low'], config.f1_paramA, config.f1_paramB)
    indicator_df['f2'] = series_from(config.f2_string, data_frame['close'], data_frame['high'], data_frame['low'], config.f2_paramA, config.f2_paramB)
    indicator_df['f3'] = series_from(config.f3_string, data_frame['close'], data_frame['high'], data_frame['low'], config.f3_paramA, config.f3_paramB)
    indicator_df['f4'] = series_from(config.f4_string, data_frame['close'], data_frame['high'], data_frame['low'], config.f4_paramA, config.f4_paramB)
    indicator_df['f5'] = series_from(config.f5_string, data_frame['close'], data_frame['high'], data_frame['low'], config.f5_paramA, config.f5_paramB)
    #print("indicator_df: \n", indicator_df)
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

def create_training_labels(src, future_candle=5):
    """
    Creates training labels based on the direction of price action.

    Parameters:
    src (DataFrame): DataFrame containing the close price data.
    future_candle (int): Number of bars to look ahead for label determination.

    Returns:
    y_train_array (array): Array of training labels.
    """
    # Check if the DataFrame has enough rows
    if len(src) < future_candle:
        raise ValueError("DataFrame is too short for the specified future_candle length.")

    # Initialize an empty list for training labels
    y_train_array = []
    
    for i in range(future_candle):
        y_train_array.append(config.Label_neutral)

    # Iterate through the DataFrame
    for i in range(future_candle, len(src)):
        #current_price = src.iloc[i]
        #future_price = src.iloc[i + future_candle]
        price_list = src.iloc[i - future_candle: i]
        #if current_price < future_price:
        #    y_train_array.append(config.Label_long)     # Price increased, label as long
        #elif current_price > future_price:
        #    y_train_array.append(config.Label_short)    # Price decreased, label as short
        #else:
        #    y_train_array.append(config.Label_neutral)  # Price stayed the same, label as neutral
        y_train_array.append(trend_direction(price_list.to_numpy()))



    
    return y_train_array

def get_lorentzian_distance(current_features, historical_features):
    """
    Calculate the Lorentzian distance between the current features and the historical features at a given index.
    """
    distance = 0
    for k in range(config.Settings_featureCount):
        # Calculate the distance for each feature
        distance += np.log(1 + abs(current_features[f'f{k+1}'] - historical_features[f'f{k+1}']))
    return distance

def approximate_nearest_neighbors(indicator_df, y_train_array):
    """
    Performs an Approximate Nearest Neighbors Search using the Lorentzian Distance.

    Parameters:
    indicator_df (DataFrame): The data frame with technical indicators.
    y_train_array (array): Array of training labels.

    Returns:
    predictions (array): The array of predictions.
    """
    predictions = []
    distances = []
    lastDistance = 999999

    maxBarsBackIndex = len(indicator_df) - config.Settings_maxBarsBack
    current_feature = indicator_df.iloc[-1]
    #print(current_feature)
    #print(current_feature)
    #print(maxBarsBackIndex)
    #print(len(indicator_df))
    #print(indicator_df)
    # Looping through each index starting from maxBarsBackIndex to the end of indicator_df.
    for historical_index in range(maxBarsBackIndex, len(indicator_df) - config.FUTURE_CANDLE):
        # Calculating the Lorentzian distance between the current feature and a previous one at index historical_index.
        distance = get_lorentzian_distance(current_feature, indicator_df.iloc[historical_index])

        # Checking two conditions:
        # 1. The modulo operation (i % 4 == 0) ensures the calculation is only done every 4 bars.
        # 2. The distance is either the first one calculated (lastDistance == -1.0) or 
        #    it is equal to or greater than the last distance noted (d >= lastDistance).
        # 3. The first config.Settings_neighborsCount calculated need to be added to distance array as they are currently closest
        if (historical_index % 4 == 0) and ((distance <= lastDistance) or (len(predictions) < config.Settings_neighborsCount)):

            # Updating lastDistance to the current calculated distance.
            lastDistance = distance

            # Appending the current distance to the distances list.
            distances.append(distance)

            # Appending the corresponding label from y_train_array to the predictions list.
            predictions.append(y_train_array[historical_index])  

            # Checking if the number of predictions exceeds the specified neighbors count.
            if len(predictions) > config.Settings_neighborsCount:
                # Updating lastDistance to the distance at the 75th percentile.
                lastDistance = distances[config.Settings_neighborsCount - round(config.Settings_neighborsCount * 0.75)]

                # Trimming the distances list to only include distances from the 75th percentile onward.
                distances = distances[1:]

                # Similarly, trimming the predictions list to align with the updated distances list.
                predictions = predictions[1:]

    #print(predictions)
    #print(distances)
    #print(historical_index)

            

    
    prediction = sum(predictions)
    return prediction

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
    res = np.empty((config.NEIGHBOURS_COUNT, future_candle))
    time = np.empty((config.NEIGHBOURS_COUNT, future_candle))

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

    X = np.arange(len(numbers)).reshape(-1, 1)  # Reshape to 2D array as required by sklearn
    y = np.array(numbers)

    model = LinearRegression().fit(X, y)
    trend = 1 if model.coef_[0] > 0 else -1 if model.coef_[0] < 0 else 0

    return trend

if __name__ == "__main__":
    # Load data from JSON file instead of fetching from NSE
    #json_filename = 'BTC_USD/BTC_USD-ONE_MINUTE-2015_8_1-to-2024_1_31.json'
    #json_filename = 'BTC_USD/BTC_USD-FIVE_MINUTE-2015_8_1-to-2024_1_31.json'
    #json_filename = 'BTC_USD/BTC_USD-FIFTEEN_MINUTE-2015_8_1-to-2024_1_31.json'
    #json_filename = 'BTC_USD/BTC_USD-ONE_HOUR-2015_8_1-to-2024_1_31.json'
    #json_filename = 'BTC_USD/BTC_USD-SIX_HOUR-2015_8_1-to-2024_1_31.json'
    json_filename = 'BTC_USD/BTC_USD-ONE_DAY-2015_8_1-to-2024_1_31.json'
    #json_filename = 'test.json'
    og_df = pd.read_json(json_filename)
    print('uisng this chart: ', json_filename)

    # Convert columns to appropriate types
    og_df['start'] = pd.to_numeric(og_df['start'])
    og_df['dateTime'] = pd.to_datetime(og_df['dateTime'])
    og_df['high'] = pd.to_numeric(og_df['high'])
    og_df['low'] = pd.to_numeric(og_df['low'])
    og_df['open'] = pd.to_numeric(og_df['open'])
    og_df['close'] = pd.to_numeric(og_df['close'])
    og_df['volume'] = pd.to_numeric(og_df['volume'])
    og_df['confidence'] = 0
    og_df['false_up'] = 0
    og_df['false_down'] = 0
    og_df['false_none'] = 0
    og_df['true_up'] = 0
    og_df['true_down'] = 0
    og_df['true_none'] = 0

    # Reverse the DataFrame to have the most recent data at the end
    og_df = og_df.iloc[::-1]
    og_df = og_df.reset_index(drop=True)
    
    # Calculate technical indicators
    indicator_df = calculate_technical_indicators(og_df)

    # Handle NaN values, e.g., fill with the first non-NaN value or another strategy (currently using backward filling)
    indicator_df.fillna(0, inplace=True)
    #print("Final Technical Indicators DataFrame:\n", indicator_df)

    y_train_array = create_training_labels(og_df['close'], config.FUTURE_CANDLE)
    
    confidence = 0

    total_false_up = 0
    total_false_down = 0
    total_true_up = 0
    total_true_down = 0
    total_no_prediction = 0
    prediction_start = config.Settings_maxBarsBack + 200
    data_start = 50

    for curr in range(prediction_start, len(indicator_df)):
        curr_indicator_df = indicator_df.iloc[data_start:curr].reset_index(drop=True)

        confidence = 0


        # Make predictions using the trained model
        prediction = approximate_nearest_neighbors(curr_indicator_df, y_train_array)
        #sys.exit()
        #print("confidence: ", prediction)
        og_df.at[curr, 'confidence'] = prediction

    for curr in range(prediction_start, len(og_df) - config.FUTURE_CANDLE):
        false_up = 0
        false_down = 0
        true_up = 0
        true_down = 0
        no_prediction = 0

        # Grab the last few 'confidence' at current index
        current_confidence = og_df['confidence'].iloc[curr]

        # Grab the next config.FUTURE_CANDLE 'close' values starting at current index in og_df
        next_closes = og_df['close'].iloc[curr:curr + config.FUTURE_CANDLE].to_numpy()
        
        # Get the trend direction of the next_closes
        future_price_trend = trend_direction(next_closes)
        #print(future_price_trend)

        # Check if our confidence matches future trend
        if current_confidence > 0:
            # trend is up and confidence is up = good
            if future_price_trend > 0:
                true_up = 1
            else:
                false_up = 1
        elif current_confidence < 0:
            # trend is up and confidence is down = bad
            if future_price_trend < 0:
                true_down = 1
            else:
                false_down = 1
        else:
            no_prediction = 1

        total_false_up += false_up
        total_false_down += false_down
        total_true_up += true_up
        total_true_down += true_down
        total_no_prediction += no_prediction
        og_df.at[curr, 'false_up'] = total_false_up
        og_df.at[curr, 'false_down'] = total_false_down
        og_df.at[curr, 'true_up'] = total_true_up
        og_df.at[curr, 'true_down'] = total_true_down
        og_df.at[curr, 'no_prediction'] = total_no_prediction

    sum = total_false_up + total_false_down + total_true_up + total_true_down
    print("total_true_up:         {} ({}%)".format(total_true_up, (total_true_up/sum)*100 ))
    print("total_false_up:        {} ({}%)".format(total_false_up, (total_false_up/sum)*100 ))
    print("total_true_down:       {} ({}%)".format(total_true_down, (total_true_down/sum)*100 ))
    print("total_false_down:      {} ({}%)".format(total_false_down, (total_false_down/sum)*100 ))
    print("wrong:                 {} ".format((total_true_up/sum)*100 + (total_true_down/sum)*100))
    print("right:                 {} ".format((total_false_up/sum)*100 + (total_false_down/sum)*100))
    print("total_no_prediction:   {} ".format(total_no_prediction))

    # Plotting
    # setting up multplie 3 plots
    figure, axis = plt.subplots(3, 1)

    # plotting real price 
    axis[0].set_title('Prediction stats')
    axis[0].plot(og_df['dateTime'], og_df['false_up'], label='false_up', color='red')
    axis[0].plot(og_df['dateTime'], og_df['false_down'], label='false_down', color='red')
    axis[0].plot(og_df['dateTime'], og_df['true_up'], label='true_up', color='green')
    axis[0].plot(og_df['dateTime'], og_df['true_down'], label='true_down', color='blue')
    axis[0].plot(og_df['dateTime'], og_df['no_prediction'], label='no_prediction', color='grey')
    axis[0].legend()

    # plotting model projected 
    #axis[1].set_title('model projected')
    #axis[1].plot(range(config.FUTURE_CANDLE), neighbors[i], label='Predicted', color='red')

    # plotting processed data 
    axis[1].set_title('volume')
    axis[1].plot(og_df['dateTime'], og_df['volume'], label='volume', color='black')
    axis[1].legend()
    #axis[1].plot(processed_df['dateTime'], processed_df['confidence'], label='confidence', color='blue')

    axis[2].set_title('Price Close')
    axis[2].plot(og_df['dateTime'], og_df['close'], label='close', color='orange')
    axis[2].legend()


    #plt.legend()
    plt.show()
