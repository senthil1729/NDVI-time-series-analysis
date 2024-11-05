import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from ndvi_ts_lstm import NDVIForecaster

class SyntheticNDVIForecaster(NDVIForecaster):
    def __init__(self, start_date, end_date, n_steps_in, n_steps_out,
                 percentile=65, bimonthly_period='2M', spline_smoothing=0.96):
        self.end_date = end_date
        self.start_date = start_date
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.percentile = percentile
        self.bimonthly_period = bimonthly_period
        self.spline_smoothing = spline_smoothing
        self.ndvi_df = None
        self.weather_df = None
        self.ndvi_interpolated = None
        self.baseline_df = None
        self.weather_df = None
        self.merged_df = None
        self.train_df = None
        self.test_df = None
        self.model_original = None
        self.model_filtered = None
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaler_y_smoothed = MinMaxScaler()
        self.current_date = pd.Timestamp.today().normalize()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = {
            'lstm_units': 244,
            'num_layers': 1, 
            'dropout_rate': 0.2887856831106061, 
            'learning_rate': 0.001827795604676652, 
            'batch_size': 128
        }

    
    def generate_synthetic_data(self, start_date, end_date):
        """Generate synthetic NDVI data patterns"""
        dates = pd.date_range(start=start_date, end=end_date, freq='5D')
        n_points = len(dates)
        
        # Calculate years and phase for annual cycle
        years = (dates - dates[0]).days / 365.25
        t = 2 * np.pi * years
        
        # Generate base NDVI with annual cycle
        base_ndvi = np.zeros(n_points)
        
        # Add varying peaks for each year
        unique_years = np.unique(dates.year)
        for year in unique_years:
            # Random variations for each year's peak
            peak_height = np.random.uniform(0.7, 0.8)    # Higher peaks
            trough_height = np.random.uniform(0.2, 0.4)  # Lower troughs
            growth_rate = np.random.uniform(0.8, 1.2)    # Varying growth rates
            decline_rate = np.random.uniform(1.5, 2.0)   # Steeper decline
            peak_shift = np.random.uniform(-0.2, 0.2)    # Timing variation
            
            # Create mask for current year
            year_mask = (dates.year == year)
            year_phase = t[year_mask] + peak_shift
            
            # Asymmetric growth-decline pattern
            growth_phase = (year_phase % (2 * np.pi)) / (2 * np.pi)
            
            # Create asymmetric pattern
            base_pattern = np.zeros(len(year_phase))
            for i, phase in enumerate(growth_phase):
                if phase < 0.4:  # Growth phase (40% of cycle)
                    x = phase / 0.4
                    base_pattern[i] = x ** (1/growth_rate)
                else:  # Decline phase (60% of cycle)
                    x = (phase - 0.4) / 0.6
                    base_pattern[i] = (1 - x) ** decline_rate
            
            # Scale pattern to NDVI range
            scaled_pattern = trough_height + (peak_height - trough_height) * base_pattern
            base_ndvi[year_mask] = scaled_pattern
        
        # Add realistic noise
        noise_amplitude = 0.02 * (base_ndvi - np.min(base_ndvi)) / (np.max(base_ndvi) - np.min(base_ndvi))
        noise = np.random.normal(0, 1, n_points) * noise_amplitude
        ndvi_values = np.clip(base_ndvi + noise, 0, 1)
        
        # Generate correlated weather data
        temp_variation = 15 * np.sin(t - np.pi/6)  # Temperature leads NDVI
        temp_min = 20 + temp_variation + np.random.normal(0, 2, n_points)
        temp_max = 30 + temp_variation + np.random.normal(0, 2, n_points)
        
        # Humidity inversely related to temperature
        humidity = 80 - temp_variation + np.random.normal(0, 5, n_points)
        
        # Precipitation with seasonal pattern
        precip_base = 10 * (1 + np.sin(t - np.pi/4))
        precipitation = np.maximum(0, precip_base * np.random.exponential(1, n_points))
        
        # Create DataFrames
        self.ndvi_df = pd.DataFrame({
            'Date': dates,
            'NDVI': ndvi_values
        })
        
        self.weather_df = pd.DataFrame({
            'Date': dates,
            'TempMin': temp_min,
            'TempMax': temp_max,
            'RelativeHumidity': humidity,
            'Precipitation': precipitation
        })

    def generate_sine_wave_data(self, start_date, end_date):
        """Generate synthetic NDVI data with a simple sine wave pattern"""
        dates = pd.date_range(start=start_date, end=end_date, freq='5D')
        n_points = len(dates)
        
        # Calculate years for annual cycle
        years = (dates - dates[0]).days / 365.25
        t = 2 * np.pi * years
        
        # Generate base NDVI with sine wave
        base_ndvi = 0.5 + 0.3 * np.sin(t)  # Oscillates between 0.2 and 0.8
        
        # Add some noise
        noise = np.random.normal(0, 0.02, n_points)
        ndvi_values = np.clip(base_ndvi + noise, 0, 1)
        
        # Generate simple weather data with sine patterns
        temp_variation = 15 * np.sin(t - np.pi/6)
        temp_min = 20 + temp_variation + np.random.normal(0, 1, n_points)
        temp_max = 30 + temp_variation + np.random.normal(0, 1, n_points)
        humidity = 80 - temp_variation + np.random.normal(0, 3, n_points)
        precipitation = 10 + 5 * np.sin(t - np.pi/4) + np.random.normal(0, 2, n_points)
        precipitation = np.maximum(0, precipitation)
        
        # Create DataFrames
        self.ndvi_df = pd.DataFrame({
            'Date': dates,
            'NDVI': ndvi_values
        })
        
        self.weather_df = pd.DataFrame({
            'Date': dates,
            'TempMin': temp_min,
            'TempMax': temp_max,
            'RelativeHumidity': humidity,
            'Precipitation': precipitation
        })

    def generate_sine_wave_data_2(self, start_date, end_date):
        """Generate synthetic data with a pure sine wave pattern"""
        dates = pd.date_range(start=start_date, end=end_date, freq='5D')
        n_points = len(dates)
        
        # Generate simple sine wave
        t = np.linspace(0, 8*np.pi, n_points)  # 2 complete cycles
        sine_wave = np.sin(t)
        
        # Scale sine wave to NDVI range (0 to 1)
        ndvi_values = (sine_wave + 1) / 2  # transforms [-1,1] to [0,1]
        
        # Create simple weather data
        temp_min = 20 + 5 * sine_wave  # oscillates between 15 and 25
        temp_max = 30 + 5 * sine_wave  # oscillates between 25 and 35
        humidity = 70 + 10 * sine_wave  # oscillates between 60 and 80
        precipitation = 5 + 5 * np.maximum(sine_wave, 0)  # only positive values
        
        # Create DataFrames
        self.ndvi_df = pd.DataFrame({
            'Date': dates,
            'NDVI': ndvi_values
        })
        
        self.weather_df = pd.DataFrame({
            'Date': dates,
            'TempMin': temp_min,
            'TempMax': temp_max,
            'RelativeHumidity': humidity,
            'Precipitation': precipitation
        })
    
    def get_ndvi_timeseries(self, start_date, end_date, data_type='synthetic'):
        """Override NDVI fetching with synthetic data generation"""
        # self.generate_synthetic_data(start_date, end_date)
        if data_type.lower() == 'sine':
            print("Generating sine wave data...")
            # self.generate_sine_wave_data(start_date, end_date)
            self.generate_sine_wave_data_2(start_date, end_date)
        else:  # default to synthetic
            print("Generating synthetic data...")
            self.generate_synthetic_data(start_date, end_date)
        
        return self.ndvi_df, self.weather_df
        # return None
    
    def extract_ndvi_data(self, ndvi_timeseries):
        """Override as we already have the NDVI DataFrame"""
        self.ndvi_df = self.apply_interpolate(self.ndvi_df)
        self.ndvi_df = self.apply_filtering(self.ndvi_df)
        self.ndvi_df = self.apply_interpolate(self.ndvi_df)
        return self.ndvi_df
    
    def get_weather_data(self, start_date, end_date):
        """Override weather data fetching as we already have synthetic data"""
        pass

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test parameters
    start_date = "2018-01-01"
    end_date = "2024-03-31"
    n_steps_in = 36
    n_steps_out = 18
    
    # Create forecaster with synthetic data
    forecaster = SyntheticNDVIForecaster(
        start_date=start_date,
        end_date=end_date,
        n_steps_in=n_steps_in,
        n_steps_out=n_steps_out
    )
    
    # Generate and prepare synthetic data
    print("Generating data...\n")
    current_date = pd.Timestamp.today().normalize()
    data_end_date = max(pd.Timestamp(end_date) + pd.DateOffset(months=3), current_date)
    
    forecaster.get_ndvi_timeseries(pd.Timestamp(start_date), data_end_date, data_type='sine')
    forecaster.extract_ndvi_data(None)
    
    print("Merging and preparing data...")
    forecaster.merge_data()
    forecaster.prepare_data()
    
    print("Training models...")
    forecaster.train_models()
    
    print("Generating forecasts...")
    test_pred_original, test_pred_smoothed, forecast_pred_original, forecast_pred_smoothed = forecaster.forecast()
    
    print("Visualizing results...")
    forecaster.visualize_results(test_pred_original, test_pred_smoothed, 
                               forecast_pred_original, forecast_pred_smoothed)

if __name__ == "__main__":
    main() 