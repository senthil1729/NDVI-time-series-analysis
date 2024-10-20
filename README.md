# NDVI Time Series Analysis and Forecasting

This project implements an NDVI (Normalized Difference Vegetation Index) time series analysis and forecasting tool using LSTM (Long Short-Term Memory). It combines satellite imagery data with weather information to predict future NDVI values for specified geographical areas.

## Features

- Fetches and processes NDVI data from Sentinel-2 satellite imagery using Google Earth Engine
- Retrieves historical weather data for the specified location
- Applies data cleaning, filtering, and smoothing techniques to NDVI time series
- Implements LSTM models for both original and smoothed NDVI data
- Provides forecasting capabilities for future NDVI values
- Visualizes historical data, predictions, and forecasts using interactive plots

## Requirements

- Python 3.7+
- Google Earth Engine account
- CUDA-capable GPU (optional, for faster training)

## Installation

1. Clone this repository:

   ```shell
   git clone https://github.com/senthilkumar-dimitra/NDVI-time-series-analysis.git
   cd ndvi-time-series-analysis
   ```

2. Install the required packages:

   ```shell
   pip install -r requirements.txt
   ```

3. Set up your [Google Earth Engine authentication](https://developers.google.com/earth-engine/guides/auth)

## Usage

Run the main script:

```shell
python ndvi_ts_lstm.py
```

## Configuration

You can modify the following parameters in the script:

- `start_date`: Start date for data retrieval
- `end_date`: End date for data retrieval
- `n_steps_in`: Number of time steps used for input sequences
- `n_steps_out`: Number of time steps to forecast
- `lstm_units`: Number of units in the LSTM layers
- `percentile`: Percentile for NDVI filtering
- `bimonthly_period`: Time interval for filtering
- `spline_smoothing`: Smoothing parameter for the spline interpolation

## Output

The script generates:

- Interactive plots showing historical NDVI data, predictions, and forecasts
- Performance metrics for the LSTM models [WIP]


