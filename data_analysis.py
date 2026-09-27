import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Create a Spark session.
spark = SparkSession.builder \
    .appName("stock_data_processing") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Define the path to the stock data folder
path = "stock_data/"

# Initialize an empty list to store sections
sections_list = []

SEQUENCE_LENGTH = 10  # number of timesteps fed into the LSTM per training example
SECTION_LENGTH = 150  # rows per training window
STEP_SIZE = 25        # stride between windows


# Function to read the data and create sections
def process_file(file_path):
    # inferSchema converts the Price column to a real number automatically,
    # since the CSV now contains plain numeric values (from yfinance) rather
    # than text -- this is what was previously causing DATATYPE_MISMATCH.
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
    df = df.withColumn("Price", F.col("Price").cast("float"))

    row_count = df.count()
    if row_count < SECTION_LENGTH:
        print(f"  Skipping {file_path}: only {row_count} rows, "
              f"need at least {SECTION_LENGTH} to build one training window.")
        return []

    window = Window.orderBy(F.monotonically_increasing_id())
    df = df.withColumn("row_num", F.row_number().over(window))

    df = df.withColumn('next5_avg', F.avg('Price').over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(1, 5)))
    df = df.withColumn('last5_avg', F.avg('Price').over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-4, -0)))

    df = df.withColumn('next10_avg', F.avg('Price').over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(1, 10)))
    df = df.withColumn('last10_avg', F.avg('Price').over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-9, -0)))

    # Moving Averages (MA)
    df = df.withColumn('ma_5', F.avg('Price').over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-4, 0)))
    df = df.withColumn('ma_10', F.avg('Price').over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-9, 0)))

    # Bollinger Bands
    rolling_std = F.stddev('Price').over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-9, 0))
    df = df.withColumn('rolling_std', rolling_std)
    df = df.withColumn('upper_band', F.col('ma_10') + 2 * rolling_std)
    df = df.withColumn('lower_band', F.col('ma_10') - 2 * rolling_std)

    # RSI
    rsi_period = 14
    price_diff = F.col('Price') - F.lag(F.col('Price'), 1).over(Window.orderBy(F.monotonically_increasing_id()))
    gain = F.when(price_diff > 0, price_diff).otherwise(0)
    loss = F.when(price_diff < 0, -price_diff).otherwise(0)
    avg_gain = F.avg(gain).over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-rsi_period + 1, 0))
    avg_loss = F.avg(loss).over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-rsi_period + 1, 0))
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # avg_loss == 0 means price only went up in that window -- RSI is defined
    # as 100 in that case (fully overbought), not an error.
    df = df.withColumn('rsi', F.when(avg_loss == 0, 100).otherwise(rsi))

    # MACD
    short_term_period = 12
    long_term_period = 26
    signal_period = 9
    ema_short = F.avg('Price').over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-short_term_period + 1, 0))
    ema_long = F.avg('Price').over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-long_term_period + 1, 0))
    macd = ema_short - ema_long
    signal_line = F.avg(macd).over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-signal_period + 1, 0))
    df = df.withColumn('macd', macd)
    df = df.withColumn('signal_line', signal_line)

    # Stochastic Oscillator
    k_period = 14
    d_period = 3
    lowest = F.min('Price').over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-k_period + 1, 0))
    highest = F.max('Price').over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-k_period + 1, 0))
    price_range = highest - lowest
    # When the price hasn't moved at all in the last 14 ticks, highest == lowest,
    # so this division would be 0/0. Instead of letting that crash the job (or
    # silently turning it into a NULL and dropping the row later, which loses
    # real data), treat a flat window as "neutral" -- 50 is the midpoint of the
    # 0-100 stochastic scale, meaning neither overbought nor oversold.
    k_values = F.when(price_range == 0, F.lit(50.0)).otherwise(
        100 * (df['Price'] - lowest) / price_range
    )
    d_values = F.avg(k_values).over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-d_period + 1, 0))
    df = df.withColumn('stochastic_k', k_values)
    df = df.withColumn('stochastic_d', d_values)

    # CCI
    cci_period = 20
    typical_price = (df['Price'] + df['Price'] + df['Price']) / 3
    mean_deviation = F.avg(F.abs(typical_price - F.avg(typical_price).over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-cci_period + 1, 0)))
                        ).over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-cci_period + 1, 0))
    # Same flat-price situation can zero out mean_deviation here too.
    cci_raw = (typical_price - F.avg(typical_price).over(Window.orderBy(F.monotonically_increasing_id()).rowsBetween(-cci_period + 1, 0))) / (0.015 * mean_deviation)
    df = df.withColumn('cci', F.when(mean_deviation == 0, F.lit(0.0)).otherwise(cci_raw))

    # remove rows with nulls (mostly the first few rows of each window, before
    # enough history has accumulated to compute the longer indicators)
    df = df.na.drop()

    indicator_columns = ['Price', 'ma_5', 'ma_10', 'rolling_std', 'upper_band', 'lower_band', 'rsi', 'macd', 'signal_line', 'stochastic_k', 'stochastic_d', 'cci']
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    assembler = VectorAssembler(inputCols=indicator_columns, outputCol="features")
    pipeline = Pipeline(stages=[assembler, scaler])
    df = pipeline.fit(df).transform(df)

    # movement_class: 'up' if the average price over the next 10 ticks is
    # higher than the average over the last 10 ticks, 'down' otherwise.
    df = df.withColumn('movement_class',
                       F.when(F.col('next10_avg') > F.col('last10_avg'), 'up')
                       .otherwise('down'))

    temp_section_list = []
    remaining_rows = df.count()
    for i in range(1, remaining_rows - SECTION_LENGTH + 2, STEP_SIZE):
        section = df.filter((F.col("row_num") >= i) & (
            F.col("row_num") < i + SECTION_LENGTH))
        temp_section_list.append(section)
    return temp_section_list


# Iterate through the files and process them
for file in os.listdir(path):
    if file.endswith(".csv"):
        file_path = os.path.join(path, file)
        sections_list.extend(process_file(file_path))

print(f"Total sections: {len(sections_list)}")

if not sections_list:
    raise RuntimeError(
        "No training sections were produced from any file in stock_data/. "
        "Check that data_scraping.py ran successfully and each CSV has at "
        f"least {SECTION_LENGTH} rows."
    )


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        # batch_first=True lets us feed tensors shaped (batch, seq_len, features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # inputs: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(inputs)
        last_step = lstm_out[:, -1, :]  # take the final timestep's hidden state
        output = self.linear(last_step)
        output = self.sigmoid(output)
        return output


indicator_columns = ['Price', 'ma_5', 'ma_10', 'rolling_std', 'upper_band',
                     'lower_band', 'rsi', 'macd', 'signal_line', 'stochastic_k', 'stochastic_d', 'cci']

# Define hyperparameters
input_dim = len(indicator_columns)
hidden_dim = 200
output_dim = 1
learning_rate = 0.01
num_epochs = 10
batch_size = 32

model = LSTMModel(input_dim, hidden_dim, output_dim)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_data, test_data = train_test_split(
    sections_list, test_size=0.2, random_state=42)


def sections_to_tensors(sections):
    """
    Pull each Spark section to the driver once (via toPandas, which is the
    standard way to hand off from a distributed feature-engineering stage to
    a PyTorch training stage) and slice it into fixed-length sequences.

    This still brings data to the driver -- for datasets that don't fit in
    driver memory, this approach needs to change (e.g. writing sequences to
    disk/Parquet and streaming them with a custom IterableDataset). For the
    dataset sizes this project is designed around, it's fine.
    """
    all_inputs = []
    all_labels = []
    for section in sections:
        pdf = section.select('scaled_features', 'movement_class').toPandas()
        section_length = len(pdf)
        for i in range(SEQUENCE_LENGTH - 1, section_length - 1):
            window_rows = pdf.iloc[i - SEQUENCE_LENGTH + 1: i + 1]
            seq = [list(v) for v in window_rows['scaled_features']]
            label = 1.0 if pdf.iloc[i + 1]['movement_class'] == 'up' else 0.0
            all_inputs.append(seq)
            all_labels.append([label])
    if not all_inputs:
        return None
    X = torch.tensor(all_inputs, dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.float32)
    return TensorDataset(X, y)


print("Collecting and windowing training sections (this pulls data to the driver once)...")
train_dataset = sections_to_tensors(train_data)
test_dataset = sections_to_tensors(test_data)

if train_dataset is None or test_dataset is None:
    raise RuntimeError("No training sequences were produced -- check that stock_data/ contains valid CSVs.")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the model using mini-batches instead of single-sample updates
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)

    avg_loss = epoch_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}')

# Evaluate the model on the test data
model.eval()
correct_predictions = 0
total_samples = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = outputs.ge(0.5).float()
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.numel()

accuracy = correct_predictions / total_samples if total_samples else 0.0
print(f'Test Accuracy: {accuracy:.4f}')

# Stop the Spark session
spark.stop()