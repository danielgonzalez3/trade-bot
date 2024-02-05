# this library is a direct translation of https://www.tradingview.com/script/e0Ek9x99-KernelFunctions/

import numpy as np

class KernelFunctions:

    @staticmethod
    def rationalQuadratic(src, lookback, relative_weight, start_at_bar):
        '''
       Rational Quadratic Kernel - An infinite sum of Gaussian Kernels of different length scales.

        @param      src             <numpy array>   The source series.
        @param      lookback        <int>           The number of bars used for the estimation. This is a sliding value that represents the most recent historical bars.
        @param      relativeWeight  <float>         Relative weighting of time frames. Smaller values resut in a more stretched out curve and larger values will result in a more wiggly curve. As this value approaches zero, the longer time frames will exert more influence on the estimation. As this value approaches infinity, the behavior of the Rational Quadratic Kernel will become identical to the Gaussian kernel.
        @param      startAtBar      <int>           Bar index on which to start regression. The first bars of a chart are often highly volatile, and omission of these initial bars often leads to a better overall fit.
        @returns    yhat            <numpy array>   The estimated values according to the Rational Quadratic Kernel.
        '''
        current_weight = 0.0
        cumulative_weight = 0.0

        # Ensure src is a numpy array for efficient computation
        src = np.array(src)

        # Iterate through the data
        for i in range(start_at_bar, len(src)):
            y = src[i]
            w = np.power(1 + (np.power(i, 2) / ((np.power(lookback, 2) * 2 * relative_weight))), -relative_weight)
            current_weight += y * w
            cumulative_weight += w

        # Calculate and return the estimated yhat value
        yhat = current_weight / cumulative_weight
        return yhat

    @staticmethod
    def gaussian(src, lookback, start_at_bar):
        '''
        Gaussian Kernel - A weighted average of the source series. The weights are determined by the Radial Basis Function (RBF).

        @param      src             <numpy array>   The source series.
        @param      lookback        <int>           The number of bars used for the estimation. This is a sliding value that represents the most recent historical bars.
        @param      start_at_bar    <int>           Bar index on which to start regression. The first bars of a chart are often highly volatile, and omission of these initial bars often leads to a better overall fit.
        @returns    yhat            <numpy array>   The estimated values according to the Gaussian Kernel.
        '''
        current_weight = 0.0
        cumulative_weight = 0.0

        # Ensure src is a numpy array for efficient computation
        src = np.array(src)

        for i in range(start_at_bar, len(src)):
            y = src[i]
            w = np.exp(-np.power(i, 2) / (2 * np.power(lookback, 2)))
            current_weight += y * w
            cumulative_weight += w

        # Calculate and return the estimated yhat value
        yhat = current_weight / cumulative_weight
        return yhat

    @staticmethod
    def periodic(src, lookback, period, start_at_bar):
        '''
        Periodic Kernel - The periodic kernel (derived by David Mackay) allows one to model functions which repeat themselves exactly.

        @param      src             <numpy array>   The source series.
        @param      lookback        <int>           The number of bars used for the estimation. This is a sliding value that represents the most recent historical bars.
        @param      period          <int>           The distance between repititions of the function.
        @param      start_at_bar    <int>           Bar index on which to start regression. The first bars of a chart are often highly volatile, and omission of these initial bars often leads to a better overall fit.
        @returns    yhat            <numpy array>   The estimated values according to the Periodic Kernel.
        '''
        current_weight = 0.0
        cumulative_weight = 0.0

        # Ensure src is a numpy array for efficient computation
        src = np.array(src)

        for i in range(start_at_bar, len(src)):
            y = src[i]
            w = np.exp(-2 * np.power(np.sin(np.pi * i / period), 2) / np.power(lookback, 2))
            current_weight += y * w
            cumulative_weight += w

        # Calculate and return the estimated yhat value
        yhat = current_weight / cumulative_weight
        return yhat

    @staticmethod
    def locallyPeriodic(src, lookback, period, start_at_bar):
        '''
        Locally Periodic Kernel - The locally periodic kernel is a periodic function that slowly varies with time. It is the product of the Periodic Kernel and the Gaussian Kernel.

        @param      src             <numpy array>   The source series.
        @param      lookback        <int>           The number of bars used for the estimation. This is a sliding value that represents the most recent historical bars.
        @param      period          <int>           The distance between repititions of the function.
        @param      start_at_bar    <int>           Bar index on which to start regression. The first bars of a chart are often highly volatile, and omission of these initial bars often leads to a better overall fit.
        @returns    yhat            <numpy array>   The estimated values according to the Locally Periodic Kernel.
        '''
        current_weight = 0.0
        cumulative_weight = 0.0

        # Ensure src is a numpy array for efficient computation
        src = np.array(src)

        for i in range(start_at_bar, len(src)):
            y = src[i]
            periodic_weight = np.exp(-2 * np.power(np.sin(np.pi * i / period), 2) / np.power(lookback, 2))
            gaussian_weight = np.exp(-np.power(i, 2) / (2 * np.power(lookback, 2)))
            w = periodic_weight * gaussian_weight
            current_weight += y * w
            cumulative_weight += w

        # Calculate and return the estimated yhat value
        yhat = current_weight / cumulative_weight
        return yhat
