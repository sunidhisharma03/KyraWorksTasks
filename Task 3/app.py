from flask import Flask, send_file, send_from_directory, jsonify
import os

app = Flask(__name__)

REPORTS_DIR = "reports"

# Root -> redirect to main report
@app.route("/")
def index():
    return report()

# Serve the main forecast report
@app.route("/report")
def report():
    report_path = os.path.join(REPORTS_DIR, "forecast_report.html")
    if os.path.exists(report_path):
        return send_file(report_path)
    return "Report not found", 404

# Serve all static files from the reports folder (plots, CSVs, etc.)
@app.route("/reports/<path:filename>")
def report_static(filename):
    file_path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(REPORTS_DIR, filename)
    return "File not found", 404

# Serve metrics as JSON
@app.route("/metrics")
def metrics():
    metrics_path = os.path.join(REPORTS_DIR, "metrics.txt")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            lines = f.readlines()
        return jsonify({
            "MAE": lines[0].split(":")[1].strip(),
            "RMSE": lines[1].split(":")[1].strip()
        })
    return "Metrics not found", 404

# Serve forecast as JSON (for AI workflows)
@app.route("/forecast")
def forecast_json():
    forecast_path = os.path.join(REPORTS_DIR, "forecast.csv")
    if os.path.exists(forecast_path):
        import pandas as pd
        df = pd.read_csv(forecast_path)
        return df.to_dict(orient="records")
    return "Forecast not found", 404


if __name__ == "__main__":
    app.run(debug=True)
