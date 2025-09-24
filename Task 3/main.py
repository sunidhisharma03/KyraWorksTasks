import os
import pandas as pd
from data_prep import load_and_prepare
from models import train_prophet
from utils import evaluate, save_plot, save_residuals_plot, save_components_plot

def run_pipeline():
    # 1. Load data
    df = load_and_prepare("train.csv", "store.csv")
    
    # 2. Train-test split (last 6 weeks for testing)
    train_df = df.iloc[:-42]
    test_df = df.iloc[-42:]
    
    # 3. Train model
    model, forecast = train_prophet(train_df, periods=42)
    
    # 4. Evaluate
    pred = forecast.iloc[-42:]["yhat"].values
    true = test_df["y"].values
    mae, rmse = evaluate(true, pred)
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.txt", "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\n")
    
    print("MAE:", mae, "RMSE:", rmse)
    
    # 5. Save plots
    save_plot(train_df, test_df, forecast)  # forecast_plot.png
    save_residuals_plot(test_df, forecast.iloc[-42:])  # residuals_plot.png
    save_components_plot(model, forecast)  # components.png
    
    # 6. Save forecast data for backend/agents
    forecast.to_csv("reports/forecast.csv", index=False)
    
    # 7. Build forecast vs actual table
    comparison = pd.DataFrame({
        "Date": test_df["ds"].values,
        "Actual": test_df["y"].values,
        "Forecast": forecast.iloc[-42:]["yhat"].values
    })
    comparison_html = comparison.to_html(index=False, classes="table table-striped", border=0)
    
    # 8. Generate detailed HTML report
    with open("reports/forecast_report.html", "w") as f:
        f.write(f"""
        <html>
        <head>
            <title>Forecast Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .table {{ border-collapse: collapse; width: 80%; margin-top: 20px; }}
                .table th, .table td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
                .table th {{ background-color: #f4f4f4; }}
                img {{ margin: 20px 0; border: 1px solid #ccc; }}
            </style>
        </head>
        <body>
            <h1>Rossmann Sales Forecast Report</h1>
            <h2>Evaluation Metrics</h2>
            <p><b>MAE:</b> {mae:.2f}</p>
            <p><b>RMSE:</b> {rmse:.2f}</p>
            
            <h2>Forecast vs Actual (last 6 weeks)</h2>
            {comparison_html}
            
            <h2>Forecast Plot</h2>
            <img src="/reports/forecast_plot.png" width="800">
            
            <h2>Residuals Plot</h2>
            <img src="/reports/residuals_plot.png" width="800">
            
            <h2>Trend & Seasonality Components</h2>
            <img src="/reports/components.png" width="800">
        </body>
        </html>
        """)
    
    print("Detailed report generated at reports/forecast_report.html")

if __name__ == "__main__":
    run_pipeline()
