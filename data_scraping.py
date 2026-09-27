import os
import time
import pandas as pd
import yfinance as yf

# How much history to pull and at what resolution. Yahoo only keeps 5-minute
# bars for the last 60 days, so 1 month is safely within that limit.
PERIOD = "1mo"
INTERVAL = "5m"

# data_analysis.py slices each stock into overlapping 150-row windows.
# If a stock comes back with fewer rows than this, that stock can't
# produce even a single window downstream -- better to catch it here
# with a clear warning than let data_analysis.py fail later.
MIN_ROWS_REQUIRED = 150


def fetch_data(stock_symbol: str, output_dir: str = "stock_data", retries: int = 3) -> bool:
    """
    Fetch intraday price history for one stock and save it as a CSV in the
    schema data_analysis.py expects: Stock, Date, Time, Price.

    Returns True if a usable file was written, False otherwise -- so the
    caller can report which stocks failed instead of the whole run stopping
    at the first network error.
    """
    os.makedirs(output_dir, exist_ok=True)

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Fetching data for {stock_symbol} (attempt {attempt}/{retries})...")
            ticker = yf.Ticker(stock_symbol)
            df = ticker.history(period=PERIOD, interval=INTERVAL)
            break
        except Exception as exc:
            last_error = exc
            print(f"  Network/API error for {stock_symbol}: {exc}")
            if attempt < retries:
                time.sleep(2 * attempt)  # back off a bit longer each retry
    else:
        print(f"  Giving up on {stock_symbol} after {retries} attempts ({last_error}).")
        return False

    if df.empty:
        print(f"  No data returned for {stock_symbol} -- check the ticker symbol.")
        return False

    if len(df) < MIN_ROWS_REQUIRED:
        print(f"  Warning: only {len(df)} rows for {stock_symbol}, "
              f"need at least {MIN_ROWS_REQUIRED} for one training window. "
              f"Saving anyway, but this stock likely won't contribute any sections.")

    df = df.reset_index()

    # yfinance intraday data has a 'Datetime' column
    out = pd.DataFrame({
        "Stock": stock_symbol,
        "Date": df["Datetime"].dt.strftime("%Y-%m-%d"),
        "Time": df["Datetime"].dt.strftime("%H:%M"),
        "Price": df["Close"],
    })

    csv_file_path = os.path.join(output_dir, f"{stock_symbol}_data.csv")
    out.to_csv(csv_file_path, index=False)
    print(f"  Saved {len(out)} rows to {csv_file_path}")
    return True


if __name__ == "__main__":
    # Indian stock tickers on Yahoo Finance have a .NS suffix for NSE
    stocks = ["RELIANCE.NS", "INFY.NS", "WIPRO.NS", "SUNPHARMA.NS", "TCS.NS"]

    succeeded, failed = [], []
    for stock in stocks:
        (succeeded if fetch_data(stock) else failed).append(stock)

    print(f"\nDone. {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed: {', '.join(failed)}")