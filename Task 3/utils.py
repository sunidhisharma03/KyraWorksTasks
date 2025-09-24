import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os

def evaluate(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = math.sqrt(mean_squared_error(true, pred))
    return mae, rmse

def save_plot(train_df, test_df, forecast, out_path="reports/forecast_plot.png"):
    os.makedirs("reports", exist_ok=True)
    
    plt.figure(figsize=(12,5))
    plt.plot(train_df["ds"], train_df["y"], label="Train")
    plt.plot(test_df["ds"], test_df["y"], label="Test")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
    plt.legend()
    plt.title("Forecast vs Actuals")
    plt.savefig(out_path)
    plt.close()

def save_residuals_plot(test_df, forecast, out_path="reports/residuals_plot.png"):
    residuals = test_df["y"].values - forecast["yhat"].values
    plt.figure(figsize=(12,5))
    plt.plot(test_df["ds"], residuals, marker="o")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals (Actual - Forecast)")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.savefig(out_path)
    plt.close()

def save_components_plot(model, forecast, out_path="reports/components.png"):
    fig = model.plot_components(forecast)
    fig.savefig(out_path)
    plt.close(fig)
