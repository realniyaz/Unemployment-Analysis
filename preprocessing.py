import pandas as pd
import numpy as np

def load_and_preprocess_data(data_path=r"C:\Users\Dell\Desktop\Unemployment analysis\global_unemployment_data.csv",
                             required_columns=[
                                 "country_name", "indicator_name", "sex", "age_group",
                                 "age_categories", "2014", "2015", "2016", "2017",
                                 "2018", "2019", "2020", "2021", "2022", "2023", "2024"
                             ]):
    """Loads and preprocesses data with specified columns.

    Args:
        data_path (str): Path to the CSV data file.
        required_columns (list): List of required column names.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV file at {data_path}. Check file format.")
        return None

    # Check if all required columns are present
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None

    # Select only the required columns
    df = df[required_columns]

    # --- Data Cleaning and Preprocessing ---

    # Convert year columns to numeric, handling non-numeric values
    year_cols = [str(year) for year in range(2014, 2025)]  # Generate year columns dynamically
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values in numeric columns (imputation with mean)
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())
        #df[col] = df[col].fillna(0) # or fill with 0

    # Convert categorical columns to lowercase and replace spaces with underscores
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    return df

if __name__ == "__main__":
    df = load_and_preprocess_data()
    if df is not None:
        print("Data loaded and preprocessed successfully.")
        print(df.head())
    else:
        print("Data loading or preprocessing failed.")