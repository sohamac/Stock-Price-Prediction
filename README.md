# Stock Price Movement Prediction

An experiment combining PySpark for technical-indicator feature engineering with a PyTorch LSTM for binary up/down price movement classification.

## What this actually does

1. **`data_scraping.py`** -- scrapes intraday price ticks for a single hardcoded stock/date from Google Finance using Selenium + `pyautogui` mouse simulation, and writes them to a CSV in `stock_data/`.
2. **`data_analysis.py`** -- the main pipeline:
   - Loads CSVs from `stock_data/` into Spark DataFrames.
   - Computes technical indicators (moving averages, Bollinger Bands, RSI, MACD, stochastic oscillator, CCI) using Spark window functions.
   - Assembles and standard-scales the indicators into a feature vector per row.
   - Labels each row `up`/`down` based on whether the next-10-tick average price is higher than the last-10-tick average.
   - Slices each stock's time series into overlapping 150-row sections (150-row window, 25-row stride).
   - Converts sections to pandas once per section (`toPandas()`) and builds fixed-length (10-timestep) sequences for an LSTM.
   - Trains a single-layer LSTM with mini-batch SGD (batch size 32) to predict the next tick's up/down movement.

## Architecture note: where Spark's job ends

Spark does the feature engineering (technical indicators via window functions) -- that part is genuinely distributed computation. Once the feature vectors are ready, the code hands off to pandas/PyTorch for sequence windowing and LSTM training, which runs on the driver. This is a normal and common split (Spark for ETL/feature engineering, single-machine deep learning for sequence modeling), **not** a distributed training job -- so "processes massive historical stock datasets" should be read as "the feature engineering step is distributed; the model training step is not."

## Known Limitations

- **Data scraping is fragile.** `data_scraping.py` hardcodes a single stock ticker and date, requires a Windows-specific ChromeDriver path, and drives the page via `pyautogui` mouse movement rather than a stable API or headless scraping approach. It's a one-off script, not a repeatable data pipeline.
- **`toPandas()` per section means this doesn't scale to datasets larger than driver memory.** Each 150-row section is pulled to the driver individually during training-data preparation. For genuinely large datasets, sequences would need to be written to disk (e.g. Parquet) and streamed via a custom `IterableDataset` instead of collected in memory up front.
- **No train/validation split for hyperparameter tuning** -- only a train/test split. No learning rate scheduling, early stopping, or hyperparameter search.
- **Single LSTM layer, fixed hidden size (200)** with no dropout or regularization -- likely to overfit on small per-stock datasets.
- **Labeling is naive:** any next-10-average strictly greater than last-10-average counts as "up," with no threshold for noise -- small fluctuations near zero are labeled the same as strong moves.

## Running it

```bash
pip install -r requirements.txt
# Requires stock_data/ to contain one or more CSVs with a `price` and `Time` column
python data_analysis.py
```

Data scraping (optional, requires Chrome + chromedriver on the system PATH, and edits to the `stock` / `date` variables at the top of the file):
```bash
python data_scraping.py
```

## License

MIT
