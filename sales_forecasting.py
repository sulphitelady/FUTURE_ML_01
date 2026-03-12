# sales_forecasting.py
# Sales & Demand Forecasting Project
# Complete version with Linear Regression & Prophet Forecast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from prophet import Prophet

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
data = pd.read_csv("superstore.csv", encoding="latin1")
data['Order Date'] = pd.to_datetime(data['Order Date'])
data = data.dropna(subset=['Sales'])

# -----------------------------
# 2️⃣ Linear Regression Forecast
# -----------------------------
# Feature engineering
data['year'] = data['Order Date'].dt.year
data['month'] = data['Order Date'].dt.month
data['day'] = data['Order Date'].dt.day

X = data[['year', 'month', 'day']]
y = data['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
lr_predictions = lr_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, lr_predictions)
r2 = r2_score(y_test, lr_predictions)

print("===== Linear Regression Model Performance =====")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization: Actual vs Predicted Sales
plt.figure(figsize=(12,6))
plt.plot(y_test.values[:100], label="Actual Sales")
plt.plot(lr_predictions[:100], label="Predicted Sales (LR)")
plt.legend()
plt.title("Linear Regression: Actual vs Predicted Sales")
plt.xlabel("Sample Index")
plt.ylabel("Sales")
plt.show()

# -----------------------------
# 3️⃣ Prophet Forecast
# -----------------------------
# Aggregate sales by month for Prophet
monthly_sales = data.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
monthly_sales.columns = ['ds', 'y']  # Prophet requires 'ds' and 'y'

# Initialize Prophet model
prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
prophet_model.fit(monthly_sales)

# Make future dataframe for next 6 months
future = prophet_model.make_future_dataframe(periods=6, freq='ME')
forecast = prophet_model.predict(future)

# Plot Prophet forecast
fig1 = prophet_model.plot(forecast)
plt.title("Prophet Sales Forecast (Next 6 Months)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# Optional: Save forecast to CSV for Power BI
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast_prophet.csv", index=False)
print("Prophet forecast saved to forecast_prophet.csv")

from pmdarima import auto_arima
import matplotlib.pyplot as plt

# Prepare data
monthly_sales.set_index('ds', inplace=True)

# Fit auto ARIMA
arima_model = auto_arima(monthly_sales['y'], seasonal=True, m=12)
arima_model.summary()

# Forecast 6 months ahead
forecast = arima_model.predict(n_periods=6)

# Plot
plt.figure(figsize=(10,5))
plt.plot(monthly_sales.index, monthly_sales['y'], label='Historical')
plt.plot(pd.date_range(monthly_sales.index[-1]+pd.offsets.MonthBegin(1), periods=6, freq='ME'), forecast, label='Forecast')
plt.title("Sales Forecast with ARIMA")
plt.legend()
plt.show()

# -----------------------------
# 4️⃣ Optional: Combine Forecasts for Dashboard
# -----------------------------
# Linear Regression monthly aggregation
lr_monthly = data.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
lr_monthly['Predicted_Sales_LR'] = lr_model.predict(
    pd.DataFrame({
        'year': lr_monthly['Order Date'].dt.year,
        'month': lr_monthly['Order Date'].dt.month,
        'day': lr_monthly['Order Date'].dt.day
    })
)

# Save combined CSV for Power BI
lr_monthly.to_csv("sales_lr_monthly.csv", index=False)
print("Linear Regression monthly forecast saved to sales_lr_monthly.csv")