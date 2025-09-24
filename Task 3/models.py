from prophet import Prophet

def train_prophet(df, periods=42):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.add_country_holidays("DE")  # Germany holidays
    model.fit(df)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast
