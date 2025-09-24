import pandas as pd

def load_and_prepare(train_path="train.csv", store_path="store.csv"):
    train = pd.read_csv(train_path, parse_dates=["Date"])
    store = pd.read_csv(store_path)
    
    # Merge
    df = train.merge(store, on="Store", how="left")
    
    # Keep only open days
    df = df[df["Open"] == 1]
    
    # Aggregate across all stores (total daily sales)
    daily = df.groupby("Date")["Sales"].sum().reset_index()
    daily = daily.rename(columns={"Date": "ds", "Sales": "y"})
    
    return daily
