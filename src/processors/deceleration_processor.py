import argparse
import pandas as pd

def processor(input_csv_path, output_csv_path):
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(input_csv_path)
    new_df = df.drop([' id', ' z', ' h', ' p', ' r', ' wheel_angle', ' wheel_rot'], axis=1)
    new_df.rename(columns=lambda x: x.strip(), inplace=True)
    new_df['name'] = new_df['name'].str.strip()

    # Create copies of the filtered DataFrames
    df_ego = new_df.loc[new_df['name'] == 'hero'].copy()
    df_lead = new_df.loc[new_df['name'] == 'adversary'].copy()

    # Drop the 'name' column from both dataframes
    df_ego.drop(columns=['name'], inplace=True)
    df_lead.drop(columns=['name'], inplace=True)

    t = 0.04  # Time interval in seconds
    car_length = 5 # Length of the car in meters

    # Rename the columns to include the name
    df_ego.rename(columns={col: f"ego_{col}" for col in df_ego.columns}, inplace=True)
    df_ego.rename(columns={'ego_time': 'time'}, inplace=True)
    df_ego['ego_acceleration'] = (df_ego['ego_x'] - 2 * df_ego['ego_x'].shift(1) + df_ego['ego_x'].shift(2)) / (t**2)
    df_ego['ego_acceleration'] = df_ego['ego_acceleration'].fillna(0)

    df_lead.rename(columns={col: f"lead_{col}" for col in df_lead.columns}, inplace=True)
    df_lead.rename(columns={'lead_time': 'time'}, inplace=True)
    df_lead['actor_acceleration'] = (df_lead['lead_x'] - 2 * df_lead['lead_x'].shift(1) + df_lead['lead_x'].shift(2)) / (t**2)
    df_lead['actor_acceleration'] = df_lead['actor_acceleration'].fillna(0)
    df_lead.rename(columns={'lead_speed': 'actor_speed'}, inplace=True)

    # Merge the dataframes on the common columns
    df_merged = pd.merge(df_ego, df_lead, on=['time'])
    # df_merged_new = df_merged.drop(['ego_x', 'ego_y', 'lead_x', 'lead_y'], axis=1)

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

