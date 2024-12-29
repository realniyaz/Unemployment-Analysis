import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

def train_model(df, target_column='value', year_columns = [str(year) for year in range(2014, 2025)]):
    """Trains a linear regression model.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        target_column (str): The name of the target column.
        year_columns (list): List of year columns (features).

    Returns:
        tuple: A tuple containing the trained model, or None if an error occurs.
    """

    if df is None or df.empty:
        print("DataFrame is empty or None. Cannot train model.")
        return None

    # Check if target column and year columns exist
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found in DataFrame.")
        return None

    missing_year_cols = set(year_columns) - set(df.columns)
    if missing_year_cols:
        print(f"Missing year columns: {missing_year_cols}")
        return None

    # Prepare the data
    X = df[year_columns]
    y = df[target_column]

    # Handle categorical features using one-hot encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
      df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
      X = df.drop(target_column, axis=1) # update X after one hot encoding
      X = X.drop(year_columns, axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    return model, X, y # Return the trained model

from preprocessing import load_and_preprocess_data
import eda

required_cols = [
    "country_name", "indicator_name", "sex", "age_group",
    "age_categories", "2014", "2015", "2016", "2017",
    "2018", "2019", "2020", "2021", "2022", "2023", "2024"
]

df = load_and_preprocess_data(required_columns=required_cols)

if df is not None:
    eda(df)
    # Reshape the DataFrame for time series analysis
    year_cols = [str(year) for year in range(2014, 2025)]
    melted_df = pd.melt(df, id_vars=['country_name','indicator_name','sex','age_group','age_categories'], value_vars=year_cols, var_name='year', value_name='value')
    melted_df['year'] = pd.to_numeric(melted_df['year'])

    model_data = melted_df.dropna() # Drop rows with NaN values

    trained_model = train_model(model_data)
    if trained_model:
        model, X, y = trained_model
        print("Model training complete.")
        # Now you can use the trained model for predictions
    else:
        print("Model training failed.")

else:
    print("Failed to load or preprocess data. Model training cannot be performed.")