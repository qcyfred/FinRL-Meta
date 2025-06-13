import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_data(data_dir):
    """Load all data files from the specified directory"""
    print("Loading data files...")
    
    # Load 5-minute market data
    market_file = os.path.join(data_dir, "000905.XSHG_5m.csv")
    market_data = pd.read_csv(market_file,)
    market_data = market_data.iloc[:, 1:]
    print(f"Loaded market data: {market_data.shape}")
    
    # Load daily replay features
    daily_replay_file = os.path.join(data_dir, "000905.XSHG_daily_replay.csv")
    daily_replay_data = pd.read_csv(daily_replay_file)
    print(f"Loaded daily replay data: {daily_replay_data.shape}")
    
    # Load intraday features
    intraday_file = os.path.join(data_dir, "000905.XSHG_intraday.csv")
    intraday_data = pd.read_csv(intraday_file)
    print(f"Loaded intraday data: {intraday_data.shape}")
    
    # Load daily labels
    labels_file = os.path.join(data_dir, "000905.XSHG_INDEX_daily_labels.csv")
    labels_data = pd.read_csv(labels_file)
    print(f"Loaded labels data: {labels_data.shape}")
    
    return market_data, daily_replay_data, intraday_data, labels_data

def preprocess_datetime_columns(market_data, daily_replay_data, intraday_data, labels_data):
    """Convert datetime columns to proper datetime format"""
    print("Processing datetime columns...")
    
    labels_data['datetime'] = pd.to_datetime(labels_data['datetime'])
    labels_data['datetime'] = labels_data['datetime'] + timedelta(hours=15)

    market_data['datetime'] = pd.to_datetime(market_data['datetime'])
    daily_replay_data['datetime'] = pd.to_datetime(daily_replay_data['datetime'])
    intraday_data['datetime'] = pd.to_datetime(intraday_data['datetime'])

    # Add date column for labels merging
    labels_data['date'] = labels_data['datetime'].dt.date
    market_data['date'] = market_data['datetime'].dt.date
    daily_replay_data['date'] = daily_replay_data['datetime'].dt.date
    intraday_data['date'] = intraday_data['datetime'].dt.date
    
    return market_data, daily_replay_data, intraday_data, labels_data

def merge_all_data(market_data, daily_replay_data, intraday_data, labels_data):
    """Merge all data on datetime"""
    print("Merging all data...")
    
    # Start with market data as base
    merged_data = market_data.copy()
    
    # Merge with daily replay data on datetime
    merged_data = pd.merge(merged_data, daily_replay_data, on='datetime', how='inner', suffixes=('', '_daily_replay'))
    print(f"After merging with daily replay: {merged_data.shape}")
    
    # Merge with intraday data on datetime
    merged_data = pd.merge(merged_data, intraday_data, on='datetime', how='inner', suffixes=('', '_intraday'))
    print(f"After merging with intraday: {merged_data.shape}")
    
    # Merge with labels on date (since labels are daily)
    merged_data = pd.merge(merged_data, labels_data, on='datetime', how='left').ffill().dropna()
    merged_data = merged_data[merged_data['datetime'] <= labels_data['datetime'].max()]

    print(f"After merging with labels: {merged_data.shape}")
    
    # Drop duplicate date columns
    cols_to_drop = [col for col in merged_data.columns if col.endswith('_daily_replay') or col.endswith('_intraday')]
    cols_to_drop.extend(['date_x', 'utility', 'date_y']) # 为什么有_x, _y ？？？
    merged_data = merged_data.drop(columns=cols_to_drop, errors='ignore')

    return merged_data

def create_finrf_data_format(data):
    """Create finrf_data.csv format with required columns"""
    print("Creating finrf_data format...")
    
    data['time'] = data['datetime']
    data['index'] = range(len(data))

    data['tic'] = '000905.XSHG'  # ticker

    # 调整列的顺序
    first_ordered = [
        'tic',
        'time',
        'index',
    ]

    other_cols = [col for col in data.columns if col not in first_ordered]

    data = data[first_ordered + other_cols]

    return data

def main():
    # Input and output directories
    input_dir = "/Users/qinchaoyi/workspace/Ultron/code/AI4Fin/FinRL-Meta/data"
    output_dir = "/Users/qinchaoyi/workspace/Ultron/code/AI4Fin/FinRL-Meta/data"
    
    # Load all data
    market_data, daily_replay_data, intraday_data, labels_data = load_data(input_dir)
    
    # Preprocess datetime columns
    market_data, daily_replay_data, intraday_data, labels_data = preprocess_datetime_columns(
        market_data, daily_replay_data, intraday_data, labels_data
    )
    
    # Merge all data
    merged_data = merge_all_data(market_data, daily_replay_data, intraday_data, labels_data)
    
    # Drop rows with missing labels
    merged_data = merged_data.dropna(subset=['label'])
    print(f"Data after dropping missing labels: {merged_data.shape}")
    
    # Save concatenated data
    output_file1 = os.path.join(output_dir, "data_concat.csv")
    merged_data.to_csv(output_file1, index=False)
    print(f"Saved concatenated data to: {output_file1}")

    # 需要的列
    selected_features = [
        'datetime', 'label', 
        'open', 'high', 'low', 'close', 'volume', 
        'adx_14', 'cci_14', 'above_ma_120d', 'ma_20d_up',
    ]
    merged_data = merged_data[selected_features]

    # Filter for label == 2
    regime2_data = merged_data[merged_data['label'] == 2].copy()
    print(f"Regime 2 data shape: {regime2_data.shape}")

    if len(regime2_data) > 0:
        # Create finrf_data format for regime 2 data
        finrf_data_format_data = create_finrf_data_format(regime2_data)
        
        # Save filtered regime 2 data
        output_file2 = os.path.join(output_dir, "filtered_regime2.csv")
        finrf_data_format_data.to_csv(output_file2, index=False)
        print(f"Saved filtered regime 2 data to: {output_file2}")
        
        # Print summary statistics
        print("\n=== Summary ===")
        print(f"Total merged records: {len(merged_data)}")
        print(f"Records with label==2: {len(regime2_data)}")
        print(f"Date range: {merged_data['datetime'].min()} to {merged_data['datetime'].max()}")
        print(f"Label distribution:\n{merged_data['label'].value_counts().sort_index()}")
    else:
        print("No data found with label == 2")

if __name__ == "__main__":
    main() 