import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """Performs exploratory data analysis on the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
    """

    if df is None or df.empty:
      print("DataFrame is empty or None. Cannot perform EDA.")
      return

    print("--- Basic Information ---")
    df.info()

    print("\n--- Descriptive Statistics ---")
    print(df.describe(include='all'))  # Include categorical columns

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Rows ---")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

    # --- Visualizations ---
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include='object').columns

    # Distribution of numeric columns
    if not numeric_cols.empty:
        print("\n--- Distribution of Numeric Columns ---")
        df[numeric_cols].hist(figsize=(15, 10))
        plt.tight_layout()
        plt.show()

    # Boxplots for numeric columns
    if not numeric_cols.empty:
        print("\n--- Boxplots of Numeric Columns ---")
        for col in numeric_cols:
            plt.figure()
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()

    # Count plots for categorical columns
    if not categorical_cols.empty:
        print("\n--- Count Plots of Categorical Columns ---")
        for col in categorical_cols:
            plt.figure(figsize=(10,6))
            sns.countplot(x=df[col])
            plt.title(f"Count Plot of {col}")
            plt.xticks(rotation=45, ha='right') # Rotate x-axis labels if needed
            plt.tight_layout()
            plt.show()

    # Example: Unemployment trends over time by country (if applicable)
    if 'country_name' in df.columns and any(str(year) in df.columns for year in range(2014, 2025)):
        print("\n--- Unemployment Trends Over Time by Country ---")
        year_cols = [str(year) for year in range(2014, 2025)]
        melted_df = pd.melt(df, id_vars=['country_name'], value_vars=year_cols, var_name='year', value_name='value')
        plt.figure(figsize=(15, 8))
        sns.lineplot(x='year', y='value', hue='country_name', data=melted_df)
        plt.xticks(rotation=45)
        plt.title('Unemployment Trends Over Time by Country')
        plt.tight_layout()
        plt.show()

    # Correlation heatmap for numeric columns
    if len(numeric_cols) > 1: # check if there are at least 2 numeric columns
      print("\n--- Correlation Heatmap ---")
      plt.figure(figsize=(10, 8))
      sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
      plt.title('Correlation Heatmap')
      plt.show()

import numpy as np
from preprocessing import load_and_preprocess_data

# Example usage in analysis.ipynb
required_cols = [
    "country_name", "indicator_name", "sex", "age_group",
    "age_categories", "2014", "2015", "2016", "2017",
    "2018", "2019", "2020", "2021", "2022", "2023", "2024"
]

df = load_and_preprocess_data(required_columns=required_cols)

if df is not None:
    perform_eda(df)
else:
    print("Failed to load or preprocess data. EDA cannot be performed.")