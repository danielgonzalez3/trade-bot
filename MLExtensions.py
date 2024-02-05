# this library is a direct translation of https://www.tradingview.com/script/ia5ozyMF-MLExtensions/

import numpy as np
import pandas as pd
import pandas_ta as ta

class MLExtensions:

    @staticmethod
    def normalize_derivative(src, quadratic_mean_length):
        '''
        Returns the smoothed hyperbolic tangent of the input series.
        
        @param      src                     <numpy array>   The input series (i.e., the first-order derivative for price).
        @param      quadratic_mean_length   <int>           The length of the quadratic mean (RMS).
        @returns	n_deriv                 <numpy array>   The normalized derivative of the input series.
        '''
        
        # Convert the input list to a numpy array for efficient computation
        src = np.array(src)
    
        # Calculate the first-order derivative (difference) of the series
        # by subtracting each element from the element two places before it
        deriv = src - np.roll(src, 2)
    
        # Squaring the differences and replacing NaN with 0
        # This is similar to Pine Script's nz function which replaces NaN with a given value
        squared_deriv = np.nan_to_num(np.power(deriv, 2))
    
        # Calculate the moving sum of the squared differences
        # This is part of computing the quadratic mean
        moving_sum = np.convolve(squared_deriv, np.ones(quadratic_mean_length), mode='valid')
    
        # Calculate the quadratic mean (Root Mean Square) of the derivative
        quadratic_mean = np.sqrt(moving_sum / quadratic_mean_length)
    
        # Extend the length of quadratic_mean to match the length of src
        # by prepending NaNs (Not-a-Number) to the beginning of the array
        quadratic_mean_full = np.hstack((np.full(quadratic_mean_length - 1, np.nan), quadratic_mean))
    
        # Normalize the derivative by dividing it by the quadratic mean
        n_deriv = deriv / quadratic_mean_full
    
        return n_deriv
    
    @staticmethod
    def normalize(src, min, max):
        '''
        Rescales a source value with an unbounded range to a target range.
        
        @param      src <numpy array>   The input series
        @param      min <float>         The minimum value of the unbounded range
        @param      max <float>         The maximum value of the unbounded range
        @returns    rtn <numpy array>   The normalized series
        '''
        # Convert the input list to a numpy array
        src = np.array(src)
    
        # Find the historic minimum and maximum values in the series
        historic_min = np.min(src)
        historic_max = np.max(src)
    
        # If any value in src is NaN, replace it with historic_min/max, respectively
        historic_min = min(np.where(np.isnan(src), historic_min, src).min(), historic_min)
        historic_max = max(np.where(np.isnan(src), historic_max, src).max(), historic_max)
    
        # Rescale the series from its historic range to the new specified range
        rtn = min + (max - min) * (src - historic_min) / max(historic_max - historic_min, 1e-9)
    
        return rtn
    
    @staticmethod
    def rescale(src, old_min, old_max, new_min, new_max):
        '''
        Rescales a source value with a bounded range to anther bounded range
        
        @param      src     <numpy array>   The input series
        @param      old_min  <float>        The minimum value of the range to rescale from
        @param      old_max  <float>        The maximum value of the range to rescale from
        @param      new_min  <float>        The minimum value of the range to rescale to
        @param      new_max  <float>        The maximum value of the range to rescale to 
        @returns    rtn     <numpy array>   The rescaled series
        '''
        # Convert the input list to a numpy array
        src = np.array(src)
    
        # Calculate the scale factor, avoiding division by zero
        scale = (old_max - old_min) if old_max - old_min > 1e-10 else 1e-10
        
        # Rescale the series from the old range to the new range
        rtn = new_min + (new_max - new_min) * (src - old_min) / scale
    
        return rtn
    
    @staticmethod
    def custom_tanh(src):
        '''
        Returns the the hyperbolic tangent of the input series. The sigmoid-like hyperbolic tangent function is used to compress the input to a value between -1 and 1.
        
        @param      src         <numpy array>   The input series (i.e., the normalized derivative).
        @returns	tanh_values <numpy array>   The hyperbolic tangent of the input series.
        '''
        # Convert the input list to a numpy array for efficient computation
        src = np.array(src)
    
        # Apply the custom tanh formula to each element in the series
        tanh_values = -1 + 2 / (1 + np.exp(-2 * src))
    
        return tanh_values
    
    @staticmethod
    def dual_pole_filter(src, lookback):
        '''
        Returns the smoothed hyperbolic tangent of the input series.
        
        @param      src         <numpy array>   The input series (i.e., the hyperbolic tangent).
        @param      lookback    <int>           The lookback window for the smoothing.
        @returns    filter      <numpy array>   The smoothed hyperbolic tangent of the input series.
        '''
        # Convert the input list to a numpy array for efficient computation
        src = np.array(src)
        
        # Initialize variables based on the provided formula
        omega = -99 * np.pi / (70 * lookback)
        alpha = np.exp(omega)
        beta = -np.power(alpha, 2)
        gamma = np.cos(omega) * 2 * alpha
        delta = 1 - gamma - beta
    
        # Initialize the filter array with NaNs
        filter = np.full_like(src, np.nan)
    
        # Apply the filter to each element in the series
        for i in range(len(src)):
            sliding_avg = 0.5 * (src[i] + src[i-1] if i > 0 else src[i])
            filter[i] = (delta * sliding_avg) + gamma * (filter[i-1] if i > 0 else 0) + beta * (filter[i-2] if i > 1 else 0)
    
        return filter
    
    @staticmethod
    def tanh_transform(src, smoothing_frequency, quadratic_mean_length):
        '''
        Returns the tanh transform of the input series.
        
        @param      src                     <numpy array>   The input series.
        @param      smoothing_frequency     <int>           The frequency for smoothing.
        @param      quadratic_mean_length   <int>           The length of the quadratic mean.
        @returns    signal                  <numpy array>   The smoothed hyperbolic tangent transform of the input series.
        '''
        normalized_deriv = MLExtensions.normalize_derivative(src, quadratic_mean_length)
        tanh_result = np.tanh(normalized_deriv)
        signal = MLExtensions.dual_pole_filter(tanh_result, smoothing_frequency)
    
        return signal
    
    @staticmethod
    def n_rsi(src, n1, n2):
        """
        Calculate the normalized RSI.
    
        @param      src             <numpy array>   The input series.
        @param      n1              <int>           The length of the RSI.
        @param      n2              <int>           The smoothing length of the EMA.
        @returns    normalized_rsi  <numpy array>   The normalized RSI.
        """
        src_series = pd.Series(src)
    
        rsi_values = ta.rsi(src_series, length=n1)
        ema_rsi = ta.ema(rsi_values, length=n2)
        normalized_rsi = rescale(ema_rsi, 0, 100, 0, 1)
    
        return normalized_rsi
    
    @staticmethod
    def n_cci(src, n1, n2):
        '''
        Returns the normalized CCI ideal for use in ML algorithms.
        
        @param      src             <numpy array>   The input series (i.e., the result of the CCI calculation).
        @param      n1              <int>           The length of the CCI.
        @param      n2              <int>           The smoothing length of the CCI.
        @returns    normalized_cci  <numpy array>   The normalized CCI.
        '''
        src_series = pd.Series(src)
        
        # Calculate the CCI using pandas_ta
        cci_values = ta.cci(src_series, length=n1)
    
        # Calculate the Exponential Moving Average (EMA) of CCI
        ema_cci_values = ta.ema(cci_values, length=n2)
    
        # Normalize the EMA of CCI to the range [0, 1]
        normalized_cci = normalize(ema_cci_values, 0, 1)
    
        return normalized_cci
    
    @staticmethod
    def n_wt(src, n1=10, n2=11):
        '''
        Returns the normalized WaveTrend Classic series ideal for use in ML algorithms.
        
        @param      src             <numpy array>   The input series (i.e., the result of the WaveTrend Classic calculation).
        @param      n1              <int>           The first smoothing length for WaveTrend Classic.
        @param      n2              <int>           The second smoothing length for the WaveTrend Classic.
        @returns    normalized_wt   <numpy array>   The normalized WaveTrend Classic series.
        '''
        src_series = pd.Series(src)
        
        # Calculate EMA1
        ema1 = ta.ema(src_series, length=n1)
        
        # Calculate EMA2
        ema2 = ta.ema(np.abs(src - ema1), length=n1)
        
        # Calculate CI
        ci = (src - ema1) / (0.015 * ema2)
        
        # Calculate WT1
        wt1 = ta.ema(ci, length=n2)
        
        # Calculate WT2
        wt2 = ta.sma(wt1, length=4)
        
        # Normalize the difference between WT1 and WT2 to the range [0, 1]
        normalized_wt = normalize(wt1 - wt2, 0, 1)
    
        return normalized_wt
    
    @staticmethod
    def n_adx(high_src, low_src, close_src, n1):
        '''
        Returns the normalized ADX ideal for use in ML algorithms.
    
        @param      high_src        <numpy array>   The input series for the high price.
        @param      low_src         <numpy array>   The input series for the low price.
        @param      close_src       <numpy array>   The input series for the close price.
        @param      n1              <int>           The length of the ADX.
        @returns    normalized_adx  <numpy array>   The normalized ADX.
        '''
        high_src_series = pd.Series(high_src)
        low_src_series = pd.Series(low_src)
        close_src_series = pd.Series(close_src)
    
        # Calculate ADX using pandas_ta
        adx = ta.adx(high_src_series, low_src_series, close_src_series, length=n1)
    
        # Normalize the ADX to the range [0, 1]
        normalized_adx = normalize(adx, 0, 1)
    
        return normalized_adx
