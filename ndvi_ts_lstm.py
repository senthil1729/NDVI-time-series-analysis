import ee
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from numpy import array, hstack
from scipy.interpolate import UnivariateSpline
import plotly.graph_objects as go
# from datetime import datetime, timedelta

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class NDVIForecaster:
    def __init__(self, coordinates, start_date, end_date,
                 n_steps_in, n_steps_out, lstm_units,
                 percentile, bimonthly_period, spline_smoothing):
        self.coordinates = coordinates
        self.end_date = end_date
        self.start_date = start_date
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.lstm_units = lstm_units
        self.percentile = percentile
        self.bimonthly_period = bimonthly_period
        self.spline_smoothing = spline_smoothing
        self.ndvi_df = None
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.model_original = None
        self.model_filtered = None

    def initialize_ee(self):
        """Google Earth Engine Authentication and Initialization"""
        ee.Authenticate()
        ee.Initialize(project='senthilkumar-dimitra')
    
    def mask_clouds(self, image):
        """Mask clouds with appropriate bands available in the satellite bands"""
        qa = image.select('QA60')
        scl = image.select('SCL')
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0)) \
                    .And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return image.updateMask(cloud_mask).divide(10000).select("B.*").copyProperties(image, ["system:time_start"])

    def calculate_ndvi(self, image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    def get_ndvi_timeseries(self, start_date, end_date):
        aoi = ee.Geometry.Polygon([self.coordinates])
        ee_start_date = start_date
        ee_end_date = end_date
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(ee_start_date, ee_end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .map(self.mask_clouds) \
            .map(self.calculate_ndvi)

        def compute_ndvi(image):
            mean_ndvi = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=10
            ).get('NDVI')
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            return ee.Feature(None, {'date': date, 'NDVI': mean_ndvi})

        ndvi_collection = collection.filterBounds(aoi).select('NDVI').map(compute_ndvi) 
        ndvi_info = ndvi_collection.getInfo()
        # print("NDVI timeseries: ", ndvi_info)
        return ndvi_info

    def extract_ndvi_data(self, ndvi_timeseries, filtering=True):
        dates = []
        ndvi_values = []
        # print("Extracting NDVI data...", ndvi_timeseries)
        for feature in ndvi_timeseries['features']:
            properties = feature['properties']
            date = properties.get('date')
            ndvi = properties.get('NDVI')
            if date and ndvi is not None:
                dates.append(date)
                ndvi_values.append(ndvi)
            
        self.ndvi_df = pd.DataFrame({'Date': pd.to_datetime(dates), 'NDVI': ndvi_values})
        self.ndvi_df = self.apply_interpolate(self.ndvi_df)
        if filtering:
            self.ndvi_df = self.apply_filtering(self.ndvi_df)
            self.ndvi_df = self.apply_interpolate(self.ndvi_df)

        print("Filtered & Interpolated NDVI data: ", self.ndvi_df.shape)
        return self.ndvi_df 

    # Weather data using NASA's Power API
    def get_weather_data(self, start_date, end_date):
        variables = 'T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR'
        # Calculate the centroid of the farm
        def calculate_centroid(coords):
            lats = [coord[0] for coord in coords]
            lons = [coord[1] for coord in coords]
            centroid_lat = sum(lats)/len(lats)
            centroid_lon = sum(lons)/len(lons)
            return (centroid_lat, centroid_lon)

        centroid = calculate_centroid(self.coordinates)
        latitude, longitude = centroid
        
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        weather_start_date = start_date.strftime('%Y%m%d')
        weather_end_date = end_date.strftime('%Y%m%d')

        api_url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={variables}&community=AG&longitude={longitude}&latitude={latitude}&format=JSON&start={weather_start_date}&end={weather_end_date}"

        response = requests.get(api_url, verify=True)
        res = response.json()

        if 'properties' in res and 'parameter' in res['properties']:
            data = res['properties']['parameter']
        else:
            raise KeyError("The expected keys ('properties' and 'parameter') are not found in the response.")

        def filter_invalid_dates(data):
            return {k: v for k, v in data.items() if not k.endswith('13')}

        t2m_max = filter_invalid_dates(data['T2M_MAX'])
        t2m_min = filter_invalid_dates(data['T2M_MIN'])
        rh2m = filter_invalid_dates(data['RH2M'])
        precip = filter_invalid_dates(data['PRECTOTCORR'])

        weather_data = {
            'Date': pd.to_datetime(list(t2m_max.keys()), format='%Y%m%d'),
            'TempMax': list(t2m_max.values()),
            'TempMin': list(t2m_min.values()),
            'RelativeHumidity': list(rh2m.values()),
            'Precipitation': list(precip.values())
        }

        self.weather_df = pd.DataFrame(weather_data)
        self.weather_df = self.weather_df[
            (self.weather_df['TempMax'] >= 0) &
            (self.weather_df['TempMin'] >= 0) &
            (self.weather_df['RelativeHumidity'] >= 0) &
            (self.weather_df['Precipitation'] >= 0)
        ]
        return self.weather_df

    def merge_data(self):
        self.merged_df = pd.merge_asof(self.ndvi_df.sort_values('Date'), self.weather_df.sort_values('Date'), on='Date', direction='nearest')
        print(self.merged_df.shape)
        return self.merged_df

    def apply_filtering(self, df):
        df['BimonthlyPeriod'] = df['Date'].dt.to_period(self.bimonthly_period)
        percentile_threshold = df.groupby('BimonthlyPeriod')['NDVI'].quantile(self.percentile/100).reset_index()
        df = df.merge(percentile_threshold, on='BimonthlyPeriod', suffixes=('', '_threshold'))
        return df[df['NDVI'] >= df['NDVI_threshold']].drop(columns=['BimonthlyPeriod', 'NDVI_threshold'])

    def apply_interpolate(self, df):
        df = df.set_index('Date')
        df = df.resample('5D').interpolate(method='linear')
        return df.reset_index() 
    
    def apply_smoothing(self, df):
        # Minimum number of points needed for cubic spline
        min_points = 4
        
        # Ensure we're working with a copy and Date is a column
        df_copy = df.copy()
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.reset_index(inplace=True)
        
        if len(df_copy) < min_points:
            # Pad the data with nearby points if we don't have enough
            pad_needed = min_points - len(df_copy)
            
            # Get some data from training set for padding
            padding_data = self.train_df.tail(pad_needed)
            
            # Combine padding data with actual data
            padded_dates = pd.concat([padding_data['Date'], df_copy['Date']])
            padded_ndvi = pd.concat([padding_data['NDVI'], df_copy['NDVI']])
            
            # Create and apply spline
            spline = UnivariateSpline(
                padded_dates.map(pd.Timestamp.toordinal),
                padded_ndvi,
                s=self.spline_smoothing
            )
            
            # Only return smoothed values for original dates
            return spline(df_copy['Date'].map(pd.Timestamp.toordinal))
        else:
            # If we have enough points, proceed as normal
            spline = UnivariateSpline(
                df_copy['Date'].map(pd.Timestamp.toordinal),
                df_copy['NDVI'],
                s=self.spline_smoothing
            )
            return spline(df_copy['Date'].map(pd.Timestamp.toordinal))
    
    def create_historical_baseline(self, forecast_dates):
        baseline_predictions = []
        
        for forecast_date in forecast_dates:
            month = forecast_date.month
            day = forecast_date.day
            
            # Use filtered and original NDVI values for historical baseline
            historical_similar = self.train_df[
                (self.train_df['Date'].dt.month == month) &
                (abs(self.train_df['Date'].dt.day - day) <= 2)
            ]
            
            if len(historical_similar) > 0:
                avg_ndvi = historical_similar['NDVI'].mean()
            else:
                month_data = self.train_df[self.train_df['Date'].dt.month == month]
                if len(month_data) > 0:
                    avg_ndvi = month_data['NDVI'].mean()
                else:
                    avg_ndvi = self.train_df['NDVI'].mean()
            
            baseline_predictions.append({
                'Date': forecast_date,
                'Historical_Avg_NDVI': avg_ndvi
            })
        
        baseline_df = pd.DataFrame(baseline_predictions)
        
        # Apply filtering
        # baseline_df = self.apply_filtering(baseline_df.rename(columns={'Historical_Avg_NDVI': 'NDVI'}))
        baseline_df = self.apply_interpolate(baseline_df)
        baseline_df = baseline_df.rename(columns={'NDVI': 'Historical_Avg_NDVI'})
        
        # Apply smoothing
        baseline_df['Historical_Avg_NDVI_Smoothed'] = self.apply_smoothing(
            baseline_df.rename(columns={'Historical_Avg_NDVI': 'NDVI'})
        )
        print('Baseline_df_FnS:', baseline_df.shape)
        return baseline_df

    def prepare_data(self):
        three_months = pd.DateOffset(months=3)
        start_date = pd.Timestamp(self.start_date)
        end_date = pd.Timestamp(self.end_date)
        
        print("Merged data date range:", self.merged_df['Date'].min(), "to", self.merged_df['Date'].max())
        
        if end_date < self.current_date - three_months:
            # Case 1: End date is more than 3 months in the past
            print("Using Case 1: Past verification with future forecast from end date")
            self.train_df = self.merged_df[
                (self.merged_df['Date'] >= start_date) & 
                (self.merged_df['Date'] <= end_date)
            ].copy()
            
            test_start = end_date + pd.Timedelta(days=1)
            test_end = min(test_start + three_months, self.current_date)
            
            print(f"Fetching test data from {test_start} to {test_end}")
            # test_ndvi_timeseries = self.get_ndvi_timeseries(test_start, test_end)
            # test_ndvi_df = self.extract_ndvi_data(test_ndvi_timeseries, filtering=False)

            # test_weather_df = self.get_weather_data(test_start, test_end)
            # self.test_df = pd.merge_asof(test_ndvi_df.sort_values('Date'), 
            #                             test_weather_df.sort_values('Date'), 
            #                             on='Date', direction='nearest')

            self.test_df = self.merged_df[
                (self.merged_df['Date'] > end_date) & 
                (self.merged_df['Date'] <= test_end)
            ].copy()
            
            print('Case 1 - Train shape', self.train_df.shape, 'Test shape:', self.test_df.shape)
            print('Test date range:', self.test_df['Date'].min(), 'to', self.test_df['Date'].max())
            
            self.forecast_dates = pd.date_range(start=test_end + pd.Timedelta(days=1), end=test_end + three_months, freq='5D')
            self.forecast_needed = True
            self.case = 1

        elif end_date >= self.current_date - three_months and end_date < self.current_date:
            # Case 2: End date is within 3 months of current date
            print("Using Case 2: Combined past verification and future forecast")
            self.train_df = self.merged_df[
                (self.merged_df['Date'] >= start_date) & 
                (self.merged_df['Date'] <= end_date)
            ].copy()
            
            test_start = end_date + pd.Timedelta(days=1)
            test_end = self.current_date
            
            print(f"Fetching test data from {test_start} to {test_end}")
            # test_ndvi_timeseries = self.get_ndvi_timeseries(test_start, test_end)
            # test_ndvi_df = self.extract_ndvi_data(test_ndvi_timeseries, filtering=False)
        
            # test_weather_df = self.get_weather_data(test_start, test_end)
            # self.test_df = pd.merge_asof(test_ndvi_df.sort_values('Date'), 
            #                             test_weather_df.sort_values('Date'), 
            #                             on='Date', direction='nearest')

            self.test_df = self.merged_df[
                (self.merged_df['Date'] > end_date) & 
                (self.merged_df['Date'] <= test_end)
            ].copy()
            
            print('Case 2 - Train shape', self.train_df.shape, 'Test shape:', self.test_df.shape)
            print('Test date range:', self.test_df['Date'].min(), 'to', self.test_df['Date'].max())
            
            self.forecast_dates = pd.date_range(start=self.current_date + pd.Timedelta(days=1), end=self.current_date + three_months, freq='5D')
            self.forecast_needed = True
            self.case = 2
        
        else:
            # Case 3: End date is current date or in the future
            print("Using Case 3: Future forecast only")
            self.train_df = self.merged_df[
                (self.merged_df['Date'] >= start_date) & 
                (self.merged_df['Date'] <= self.current_date)
            ].copy()

            print('Case 3 - Train shape', self.train_df.shape)
            
            self.forecast_dates = pd.date_range(start=self.current_date + pd.Timedelta(days=1), end=self.current_date + three_months, freq='5D')
            self.forecast_needed = True
            self.case = 3

        # Apply smoothing to train data
        self.train_df['NDVI_Smoothed'] = self.apply_smoothing(self.train_df)

        if self.test_df is not None:
            self.test_df['NDVI_Smoothed'] = self.apply_smoothing(self.test_df)
        
        # Create historical baseline
        self.baseline_df = self.create_historical_baseline(self.forecast_dates)

        # Ensure continuous dates
        all_dates = pd.date_range(start=start_date, end=self.forecast_dates[-1], freq='5D')
        self.all_data = pd.DataFrame({'Date': all_dates})
        self.all_data = pd.merge_asof(self.all_data, self.train_df, on='Date', direction='nearest')

        if self.test_df is not None and not self.test_df.empty:
            self.all_data = pd.merge_asof(self.all_data, self.test_df, on='Date', direction='nearest', suffixes=('', '_test'))
        else:
            self.all_data = pd.merge_asof(self.all_data, self.baseline_df, on='Date', direction='nearest')

        print("Final data shapes:")
        print("Train data:", self.train_df.shape)
        # print("Train data:\n", self.train_df)
        if self.test_df is None:
            print("Baseline data:", self.baseline_df.shape)
        else:
            print("Test data:", self.test_df.shape)
        print("All data:", self.all_data.shape)

    def scale_data(self):
        X_scaled = self.scaler_x.fit_transform(self.train_df[['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']])
        y_scaled = self.scaler_y.fit_transform(self.train_df[['NDVI']])
        y_smoothed_scaled = self.scaler_y_smoothed.fit_transform(self.train_df[['NDVI_Smoothed']])

        train_data = hstack((X_scaled, y_scaled))
        smoothed_data = hstack((X_scaled, y_smoothed_scaled))

        return train_data, smoothed_data

    @staticmethod
    def split_sequences(sequences, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequences)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(sequences):
                break
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    def create_model(self, n_features):
        return LSTMModel(n_features, self.lstm_units, num_layers=2, output_size=self.n_steps_out).to(self.device)

    def train_models(self):
        train_data, smoothed_data = self.scale_data()
        X_train, y_train = self.split_sequences(train_data, self.n_steps_in, self.n_steps_out)
        X_smoothed, y_smoothed = self.split_sequences(smoothed_data, self.n_steps_in, self.n_steps_out)

        n_features = X_train.shape[2]

        self.model_original = self.create_model(n_features)
        self.model_filtered = self.create_model(n_features)

        self.train_model(self.model_original, X_train, y_train, "Original")
        self.train_model(self.model_filtered, X_smoothed, y_smoothed, "Filtered")

    def train_model(self, model, X, y, model_name):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        print(f"Training {model_name} model on {self.device}")

        epochs = 200
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'{model_name} Model - Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')

    def predict_future(self, model, input_sequence):
        model.eval()
        with torch.no_grad():
            predictions = []
            for i in range(0, len(input_sequence) - self.n_steps_in, self.n_steps_out):
                input_seq = torch.FloatTensor(input_sequence[i:i+self.n_steps_in]).unsqueeze(0).to(self.device)
                prediction = model(input_seq).cpu().numpy()
                predictions.extend(prediction[0])
        return predictions
    
    def get_future_weather(self, future_dates):
        historical_weather = self.weather_df.copy()
        future_weather_list = []
        
        for future_date in future_dates:
            month = future_date.month
            day = future_date.day
            
            # Use similar date range for historical weather
            historical_similar = historical_weather[
                (historical_weather['Date'].dt.month == month) &
                (abs(historical_weather['Date'].dt.day - day) <= 2)
            ]
            
            if len(historical_similar) > 0:
                avg_weather = historical_similar.mean()
            else:
                month_data = historical_weather[historical_weather['Date'].dt.month == month]
                if len(month_data) > 0:
                    avg_weather = month_data.mean()
                else:
                    avg_weather = historical_weather.mean()
            
            future_weather_list.append({
                'Date': future_date,
                'TempMin': avg_weather['TempMin'],
                'TempMax': avg_weather['TempMax'],
                'RelativeHumidity': avg_weather['RelativeHumidity'],
                'Precipitation': avg_weather['Precipitation']
            })
        
        future_weather = pd.DataFrame(future_weather_list)
        return future_weather

    def forecast(self):
        test_pred_original = None
        test_pred_smoothed = None
        forecast_pred_original = None
        forecast_pred_smoothed = None

        try:
            if self.case in [1, 2]:
                if self.test_df is not None and not self.test_df.empty:
                    test_X = self.scaler_x.transform(self.test_df[['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']])
                    last_train_data = self.train_df[['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']].tail(self.n_steps_in).values
                    last_train_X = self.scaler_x.transform(last_train_data)
                    test_input_sequence = np.vstack([last_train_X, test_X])
                    
                    test_pred_original = self.predict_future(self.model_original, test_input_sequence)
                    test_pred_smoothed = self.predict_future(self.model_filtered, test_input_sequence)
                    
                    if len(test_pred_original) == 0 or len(test_pred_smoothed) == 0:
                        print("Warning: No test predictions generated. Check your test data and model.")
                else:
                    print("No test data available for forecasting.")
            
            if self.case in [2, 3]:
                future_weather = self.get_future_weather(self.forecast_dates)
                future_X = self.scaler_x.transform(future_weather[['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']])
                
                last_known_data = self.train_df[['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']].tail(self.n_steps_in).values
                last_known_X = self.scaler_x.transform(last_known_data)
                
                forecast_input_sequence = np.vstack([last_known_X, future_X])
                
                forecast_pred_original = self.predict_future(self.model_original, forecast_input_sequence)
                forecast_pred_smoothed = self.predict_future(self.model_filtered, forecast_input_sequence)
                
                if len(forecast_pred_original) == 0 or len(forecast_pred_smoothed) == 0:
                    print("Warning: No forecast predictions generated. Check your forecast data and model.")

            # Inverse transform predictions only if they are not empty
            if test_pred_original is not None and len(test_pred_original) > 0:
                test_pred_original = self.scaler_y.inverse_transform(np.array(test_pred_original).reshape(-1, 1))
            if test_pred_smoothed is not None and len(test_pred_smoothed) > 0:
                test_pred_smoothed = self.scaler_y_smoothed.inverse_transform(np.array(test_pred_smoothed).reshape(-1, 1))
                # Apply smoothing to test predictions
                test_dates = self.test_df['Date'][:len(test_pred_smoothed)]
                test_pred_smoothed = self.apply_smoothing(pd.DataFrame({'Date': test_dates, 'NDVI': test_pred_smoothed.flatten()}))
            
            if forecast_pred_original is not None and len(forecast_pred_original) > 0:
                forecast_pred_original = self.scaler_y.inverse_transform(np.array(forecast_pred_original).reshape(-1, 1))
            if forecast_pred_smoothed is not None and len(forecast_pred_smoothed) > 0:
                forecast_pred_smoothed = self.scaler_y_smoothed.inverse_transform(np.array(forecast_pred_smoothed).reshape(-1, 1))
                # Apply smoothing to forecast predictions
                forecast_dates = pd.date_range(start=self.forecast_dates[0], periods=len(forecast_pred_smoothed), freq='5D')
                forecast_pred_smoothed = self.apply_smoothing(pd.DataFrame({'Date': forecast_dates, 'NDVI': forecast_pred_smoothed.flatten()}))
            
        except Exception as e:
            print(f"An error occurred during forecasting: {str(e)}")
            print("Debugging information:")
            print(f"Case: {self.case}")
            print(f"Train data shape: {self.train_df.shape}")
            if self.test_df is not None:
                print(f"Test data shape: {self.test_df.shape}")
            print(f"Forecast dates: {self.forecast_dates}")
        
        return test_pred_original, test_pred_smoothed, forecast_pred_original, forecast_pred_smoothed

    def visualize_results(self, test_pred_original, test_pred_smoothed, 
                      forecast_pred_original, forecast_pred_smoothed):
        fig = go.Figure()
        
        # Training data
        fig.add_trace(go.Scatter(x=self.train_df['Date'], y=self.train_df['NDVI'], 
                                mode='lines+markers', name='NDVI (Filtered+Interpolated)', 
                                line=dict(color='green')))
        fig.add_trace(go.Scatter(x=self.train_df['Date'], y=self.train_df['NDVI_Smoothed'], 
                                mode='lines', name='NDVI (Smoothed)', 
                                line=dict(color='darkseagreen')))
        
        # Test data and predictions
        if self.case in [1, 2] and self.test_df is not None and not self.test_df.empty:
            # Ensure test period is exactly 3 months
            test_start = self.train_df['Date'].max() + pd.Timedelta(days=1)
            test_end = test_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
            test_mask = (self.test_df['Date'] >= test_start) & (self.test_df['Date'] <= test_end)
            
            fig.add_trace(go.Scatter(x=self.test_df.loc[test_mask, 'Date'], y=self.test_df.loc[test_mask, 'NDVI'], 
                                    mode='lines+markers', name='Test NDVI (Actual)', 
                                    line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=self.test_df.loc[test_mask, 'Date'], y=self.test_df.loc[test_mask, 'NDVI_Smoothed'], 
                                    mode='lines', name='Test NDVI (Smoothed)', 
                                    line=dict(color='lightblue')))
            if test_pred_original is not None and len(test_pred_original) > 0:
                fig.add_trace(go.Scatter(x=self.test_df.loc[test_mask, 'Date'][:len(test_pred_original)], y=test_pred_original.flatten(),
                                        mode='lines', name='Test NDVI Predicted (Filtered+Interpolated)', 
                                        line=dict(color='orange', dash='dot')))
            if test_pred_smoothed is not None and len(test_pred_smoothed) > 0:
                fig.add_trace(go.Scatter(x=self.test_df.loc[test_mask, 'Date'][:len(test_pred_smoothed)], y=test_pred_smoothed.flatten(),
                                        mode='lines', name='Test NDVI Predicted (Smoothed)', 
                                        line=dict(color='red', dash='dot')))
            if self.case == 2:
                # Historical baseline
                # all_dates = pd.date_range(start=self.train_df['Date'].min(), end=self.forecast_dates[-1], freq='5D')
                # self.baseline_df = self.create_historical_baseline(all_dates)
                fig.add_trace(go.Scatter(x=self.baseline_df['Date'], y=self.baseline_df['Historical_Avg_NDVI'], 
                                        mode='lines+markers', name='Avg Historical Baseline (Filtered+Interpolated)', 
                                        line=dict(color='orange', dash='dash')))
                fig.add_trace(go.Scatter(x=self.baseline_df['Date'], y=self.baseline_df['Historical_Avg_NDVI_Smoothed'], 
                                        mode='lines', name='Avg Historical Baseline (Smoothed)', 
                                        line=dict(color='purple', dash='dash')))
        
        # Forecast
        if self.case == 3:
            if forecast_pred_original is not None and len(forecast_pred_original) > 0:
                forecast_dates = pd.date_range(start=self.forecast_dates[0], periods=len(forecast_pred_original), freq='5D') # periods=len(forecast_pred_original)
                fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_pred_original.flatten(), 
                                        mode='lines', name='LSTM Forecast (Filtered+Interpolated)', 
                                        line=dict(color='red', dash='dot')))
            if forecast_pred_smoothed is not None and len(forecast_pred_smoothed) > 0:
                forecast_dates = pd.date_range(start=self.forecast_dates[0], periods=len(forecast_pred_smoothed), freq='5D')  # periods=len(forecast_pred_smoothed)
                fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_pred_smoothed.flatten(), 
                                        mode='lines', name='LSTM Forecast (Smoothed)', 
                                        line=dict(color='rosybrown', dash='dot')))
        
            # Historical baseline
            # all_dates = pd.date_range(start=self.train_df['Date'].min(), end=self.forecast_dates[-1], freq='5D')
            all_dates = pd.date_range(start=self.forecast_dates[0], periods=len(forecast_pred_original), freq='5D') # ranges upto 6 months
            self.baseline_df = self.create_historical_baseline(all_dates)
            fig.add_trace(go.Scatter(x=self.baseline_df['Date'], y=self.baseline_df['Historical_Avg_NDVI'], 
                                    mode='lines+markers', name='Avg Historical Baseline (Filtered+Interpolated)', 
                                    line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(x=self.baseline_df['Date'], y=self.baseline_df['Historical_Avg_NDVI_Smoothed'], 
                                    mode='lines', name='Avg Historical Baseline (Smoothed)', 
                                    line=dict(color='purple', dash='dash')))

        fig.update_layout(
            title=f"NDVI Analysis - Case {self.case}",
            xaxis_title="Date",
            yaxis_title="NDVI",
            legend_title="Legend",
            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="right", x=1)
        )
        fig.show()

def main():
    print("Enter coordinates in the format 'lat1,lon1 lat2,lon2 lat3,lon3 lat4,lon4':")
    # Farm 1 coordinates - 11.9339692,9.6092636 11.9378834,9.607271 11.937092,9.590627 11.9297,9.591299 
    # Farm 2 coordinates - 11.876698,9.611602 11.885893,9.611618 11.885875,9.600578 11.876751,9.600584
    print("Farm-1: 11.9339692,9.6092636 11.9378834,9.607271 11.937092,9.590627 11.9297,9.591299","\nFarm-2: 11.876698,9.611602 11.885893,9.611618 11.885875,9.600578 11.876751,9.600584")
    coord_input = input().strip()
    coord_pairs = [pair.split(',') for pair in coord_input.split()]
    coordinates = [(float(lat), float(lon)) for lat, lon in coord_pairs]
    
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    
    n_steps_in = int(input("Enter steps in (default=36): ") or 36)
    n_steps_out = int(input("Enter steps out (default=18): ") or 18)
    lstm_units = int(input("Enter number of LSTM units (default=50): ") or 50)
    percentile = int(input("Enter percentile for filtering (default=65): ") or 65)
    bimonthly_period = input("Enter time interval for filtering in months (default=2): ") or '2'
    bimonthly_period = f"{bimonthly_period}M"
    spline_smoothing = float(input("Enter spline smoothing parameter (default=0.45): ") or 0.45)
    
    # Create forecaster with user inputs
    forecaster = NDVIForecaster(
        coordinates=coordinates,
        end_date=end_date,
        start_date=start_date,
        n_steps_in=n_steps_in,
        n_steps_out=n_steps_out,
        lstm_units=lstm_units,
        percentile=percentile,
        bimonthly_period=bimonthly_period,
        spline_smoothing=spline_smoothing
    )
    
    print("Initializing Earth Engine...")
    forecaster.initialize_ee()

    print("Calculating data range...")
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    current_date = pd.Timestamp.today().normalize()
    data_end_date = max(end_date + pd.DateOffset(months=3), current_date)

    print("Fetching NDVI data...")
    ndvi_timeseries = forecaster.get_ndvi_timeseries(start_date, data_end_date)
    forecaster.extract_ndvi_data(ndvi_timeseries)

    print("Fetching weather data...")
    forecaster.get_weather_data(start_date, data_end_date)
    
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

