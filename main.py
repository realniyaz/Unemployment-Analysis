import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_and_preprocess_data
import eda
import model_trainig

def main():
    """Main function to run the unemployment analysis project."""

    required_cols = [
        "country_name", "indicator_name", "sex", "age_group",
        "age_categories", "2014", "2015", "2016", "2017",
        "2018", "2019", "2020", "2021", "2022", "2023", "2024"
    ]

    df = load_and_preprocess_data(required_columns=required_cols)

    if df is None:
        print("Failed to load or preprocess data. Exiting.")
        return  # Exit the program

    eda(df)

    # Reshape the DataFrame for time series analysis (if needed for your model)
    year_cols = [str(year) for year in range(2014, 2025)]
    try:
        melted_df = pd.melt(df, id_vars=['country_name','indicator_name','sex','age_group','age_categories'], value_vars=year_cols, var_name='year', value_name='value')
        melted_df['year'] = pd.to_numeric(melted_df['year'])
        model_data = melted_df.dropna() # Drop rows with NaN values
    except KeyError as e:
        print(f"Error during melting: {e}. Check if the columns exist.")
        return

    trained_model = model_trainig(model_data)

    if trained_model:
        model, X, y = trained_model
        print("Model training complete.")

        # Example: Make predictions (you'll need to prepare your input data for prediction)
        # new_data = pd.DataFrame(...) # Create a DataFrame with the same features as X
        # predictions = model.predict(new_data)
        # print("Predictions:", predictions)

    else:
        print("Model training failed.")

if __name__ == "__main__":
    main()