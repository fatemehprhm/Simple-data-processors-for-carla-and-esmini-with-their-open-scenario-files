import argparse
import pandas as pd
import numpy as np

def processor(input_csv_path, output_csv_path):
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(input_csv_path)
    new_df = df.drop([' id', ' z', ' h', ' p', ' r', ' wheel_angle', ' wheel_rot'], axis=1)
    new_df.rename(columns=lambda x: x.strip(), inplace=True)
    new_df['name'] = new_df['name'].str.strip()

    # Create copies of the filtered DataFrames
    df_ego = new_df.loc[new_df['name'] == 'ego'].copy()
    df_overtaker = new_df.loc[new_df['name'] == 'overtaker'].copy()

    # Drop the 'name' column from both dataframes
    df_ego.drop(columns=['name'], inplace=True)
    df_overtaker.drop(columns=['name'], inplace=True)

    t = 0.04  # Time interval in seconds
    car_length = 5.04
    car_width = 2
    lane_length = 1.75

    # Rename the columns to include the name
    df_ego.rename(columns={col: f"ego_{col}" for col in df_ego.columns}, inplace=True)
    df_ego.rename(columns={'ego_time': 'time'}, inplace=True)
    df_ego['ego_acc_x'] = (df_ego['ego_x'] - 2 * df_ego['ego_x'].shift(1) + df_ego['ego_x'].shift(2)) / (t**2)
    df_ego['ego_acc_x'] = df_ego['ego_acc_x'].fillna(0)

    df_overtaker.rename(columns={col: f"overtaker_{col}" for col in df_overtaker.columns}, inplace=True)
    df_overtaker.rename(columns={'overtaker_time': 'time'}, inplace=True)
    df_overtaker.rename(columns={'overtaker_speed': 'actor_speed'}, inplace=True)
    df_overtaker['actor_acc_x'] = (df_overtaker['overtaker_x'] - 2 * df_overtaker['overtaker_x'].shift(1) + df_overtaker['overtaker_x'].shift(2)) / (t**2)
    df_overtaker['actor_acc_x'] = df_overtaker['actor_acc_x'].fillna(0)
    df_overtaker['actor_acc_y'] = (df_overtaker['overtaker_y'] - 2 * df_overtaker['overtaker_y'].shift(1) + df_overtaker['overtaker_y'].shift(2)) / (t**2)
    df_overtaker['actor_acc_y'] = df_overtaker['actor_acc_y'].fillna(0)

    # Merge the dataframes on the common columns
    df_merged = pd.merge(df_ego, df_overtaker, on=['time'])

    # df_merged_new = df_merged.drop(['ego_x', 'ego_y', 'overtaker_x', 'overtaker_y'], axis=1)

    # Save the updated dataframe to the output CSV file
    df_merged.to_csv(output_csv_path, index=False)
    print("CSV file created successfully.")

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Processing CSV file.")
    parser.add_argument("--csv", help="Input CSV file path")
    parser.add_argument("--output", help="Output CSV file name")

    args = parser.parse_args()

    # Call the function to process the csv file
    processor(args.csv, args.output)

