# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.title(" Stock Market Predicting App")


uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.lower() for col in df.columns]

    # Check for required columns
    if 'date' not in df.columns or 'value' not in df.columns:
        st.error("CSV must contain 'date' and 'value' columns.")
    else:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
        st.write("Preview:", df.head())

        # Choose decomposition type
        decomposition_type = st.radio("Select Decomposition Type", ["Additive", "Multiplicative"])
        period = st.number_input("Seasonal Period (e.g., 12 for monthly)", min_value=2, value=12)

        # Decompose
        result = seasonal_decompose(df['value'], model=decomposition_type.lower(), period=period)
        st.subheader("Decomposition")
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        result.observed.plot(ax=axs[0], title='Observed')
        result.trend.plot(ax=axs[1], title='Trend')
        result.seasonal.plot(ax=axs[2], title='Seasonal')
        result.resid.plot(ax=axs[3], title='Residuals')
        st.pyplot(fig)

        # Model selection
        model_choice = st.selectbox("Choose a forecasting model", ["ARIMA", "ETS", "Prophet"])
        forecast_steps = st.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30)

        # Train-test split
        train_size = int(len(df) * 0.8)
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]

        # Forecasting
        if model_choice == "ARIMA":
            model = ARIMA(train['value'], order=(5, 1, 0)).fit()
            forecast = model.forecast(steps=len(test))

        elif model_choice == "ETS":
            model = ExponentialSmoothing(train['value'], trend='add', seasonal='add', seasonal_periods=period).fit()
            forecast = model.forecast(steps=len(test))

        elif model_choice == "Prophet":
            prophet_df = train.reset_index().rename(columns={'date': 'ds', 'value': 'y'})
            model = Prophet(daily_seasonality=True)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=len(test))
            forecast_df = model.predict(future)
            forecast = forecast_df.set_index('ds')['yhat'].iloc[-len(test):]

        # Evaluation
        rmse = np.sqrt(mean_squared_error(test['value'], forecast))
        mae = mean_absolute_error(test['value'], forecast)
        mape = np.mean(np.abs((test['value'] - forecast) / np.maximum(test['value'], 1))) * 100

        st.subheader("Forecast vs Actual")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(test.index, test['value'], label='Actual', color='blue')
        ax2.plot(test.index, forecast, label='Forecast', color='green')
        ax2.set_title(f"{model_choice} Forecast")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("Evaluation Metrics")
        st.write(f"**RMSE**: {rmse:.2f}")
        st.write(f"**MAE**: {mae:.2f}")
        st.write(f"**MAPE**: {mape:.2f}%")
