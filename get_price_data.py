import json, os
from datetime import datetime, timezone
from coinbaseadvanced.client import CoinbaseAdvancedTradeAPIClient, Granularity
import config
# Importing necessary libraries: json for handling JSON data,
# datetime and timezone for date and time manipulation,
# and classes from coinbaseadvanced.client for Coinbase Pro API interaction.

def save_to_json_file(candles, file_name, directory):
    """
    Saves candle data to a JSON file in a specified directory.

    Parameters:
    candles (list): A list of candle objects to be saved.
    file_name (str): The name of the file to save the data in.
    directory (str): The directory to save the file in.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    full_path = os.path.join(directory, file_name)
    with open(full_path, "w") as f:
        json_candles = [candle_to_dict(candle) for candle in candles]
        json.dump(json_candles, f, indent=4)  # Added indentation for readability


def candle_to_dict(candle):
    """
    Converts a candle object to a dictionary.

    Parameters:
    candle (object): The candle object to convert.

    Returns:
    dict: A dictionary representation of the candle object.
    """
    # Converting the candle object attributes to a dictionary
    return {
        "start":    candle.start,
        "dateTime": datetime.utcfromtimestamp(int(candle.start)).strftime('%B %d, %Y %H:%M'),
        "high":     candle.high,
        "low":      candle.low,
        "open":     candle.open,
        "close":    candle.close,
        "volume":   candle.volume
    }

if __name__ == "__main__":
    # Setting up the API client with your credentials
    try:
        client = CoinbaseAdvancedTradeAPIClient(api_key=PUBLIC, secret_key=SECRET)
    except Exception as e:
        print(f"Error setting up Coinbase API client: {e}")
        exit(1)

    # User input for ticker symbol
    ticker_symbol = input("Enter the ticker symbol (e.g., BTC-USD): ")

    # Create a directory name based on the ticker symbol
    directory_name = ticker_symbol.replace('-', '_')

    # List of tuples with Granularity objects and their string representations
    timeframes = [
        #(Granularity.ONE_MINUTE,    'ONE_MINUTE'),
        #(Granularity.FIVE_MINUTE,   'FIVE_MINUTE'),
        (Granularity.FIFTEEN_MINUTE,'FIFTEEN_MINUTE'),
        (Granularity.THIRTY_MINUTE, 'THIRTY_MINUTE'),
        (Granularity.ONE_HOUR,      'ONE_HOUR'),
        (Granularity.TWO_HOUR,      'TWO_HOUR'),
        (Granularity.SIX_HOUR,      'SIX_HOUR'),
        (Granularity.ONE_DAY,       'ONE_DAY')
    ]

    # Looping through each timeframe to fetch and save candle data
    for gran, timeframe in timeframes:
        print(f'Starting timeframe {timeframe} for {ticker_symbol} at {datetime.now()}')

        try:
            # Fetching product candles data from Coinbase Pro API
            res = client.get_product_candles_all(ticker_symbol,
                        start_date=datetime(2015, 8, 1, tzinfo=timezone.utc),
                        end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
                        granularity=gran)
            
            # File name for the JSON data
            file_name = f'{directory_name}-{timeframe}-2015_8_1-to-2024_1_31.json'

            # Saving the fetched data to a JSON file in the specified directory
            try:
                save_to_json_file(res, file_name, directory_name)
            except Exception as e:
                print(f"Error saving data to file {file_name}: {e}")

        except Exception as e:
            print(f"Error fetching data for timeframe {timeframe}: {e}")

        print(f'Finished grabbing timeframe {timeframe} for {ticker_symbol} at {datetime.now()}')
