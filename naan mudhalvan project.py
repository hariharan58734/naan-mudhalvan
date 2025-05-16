import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from IPython.display import display, Markdown

# --- Data Loading ---
print("--- Data Loading ---")
try:
    file_path = '100 data excel.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    print(f"Successfully loaded data from {file_path}")
    display(df.head())
    print(f"DataFrame shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: '{file_path}' not found. Please ensure the file is in the current working directory or provide the full path.")
    df = None
print("\n")

if df is not None:
    # --- Data Exploration ---
    print("--- Data Exploration ---")
    print("Data Types:\n", df.dtypes)
    print("\nSummary Statistics:\n", df.describe(include='all'))
    print("\nDataFrame Info:\n")
    df.info()
    print("\nMissing Value Percentage:\n", df.isnull().sum() / len(df) * 100)

    for col in ['Label', 'Source', 'Language']:
        if col in df.columns:
            print(f"\nUnique values in {col}:\n{df[col].unique()}")
            print(f"\nValue counts in {col}:\n{df[col].value_counts()}")
        else:
            print(f"\nColumn '{col}' not found in DataFrame.")

    print("\nNumber of duplicate rows:", df.duplicated().sum())
    print("\n")

    # --- Data Cleaning ---
    print("--- Data Cleaning ---")
    print("\nMissing Value Percentage (after potential handling, if any):\n", df.isnull().sum() / len(df) * 100)
    if 'Language' in df.columns:
        print("\nUnique values in Language:\n", df['Language'].unique())
    else:
        print("\nColumn 'Language' not found in DataFrame.")
    print(f"\nNumber of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
    display(df.head())
    print("\n")

    # --- Data Preparation ---
    print("--- Data Preparation ---")
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            print("Successfully converted 'Date' column to datetime.")
            # Handle potential NaT values if 'coerce' was used
            if df['Date'].isnull().sum() > 0:
                print(f"Warning: {df['Date'].isnull().sum()} invalid dates found and converted to NaT.")
        except ValueError as e:
            print(f"Error converting 'Date' column: {e}")
    else:
        print("\nColumn 'Date' not found in DataFrame. Skipping datetime conversion.")

    if 'Content' in df.columns:
        df['Content_Length'] = df['Content'].astype(str).apply(len)
        print("Created 'Content_Length' column.")
    else:
        print("\nColumn 'Content' not found in DataFrame. Skipping 'Content_Length' creation.")

    display(df.head())
    print(df.dtypes)
    print("\n")

    # --- Data Analysis ---
    print("--- Data Analysis ---")
    numerical_cols = ['ID', 'Content_Length']
    present_numerical_cols = [col for col in numerical_cols if col in df.columns]
    if present_numerical_cols:
        print(df[present_numerical_cols].describe())

        if 'Content_Length' in df.columns:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.hist(df['Content_Length'].dropna(), bins=20, color='skyblue', edgecolor='black')
            plt.title('Distribution of Content Length')
            plt.xlabel('Content Length')
            plt.ylabel('Frequency')

            plt.subplot(1, 2, 2)
            plt.boxplot(df['Content_Length'].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
            plt.title('Box Plot of Content Length')
            plt.xlabel('Content Length')
            plt.tight_layout()
            plt.show()

            if 'Label' in df.columns:
                 print("\nContent Length by Label:")
                 print(df.groupby('Label')['Content_Length'].describe())
                 plt.figure(figsize=(8, 6))
                 df.boxplot(column='Content_Length', by='Label', patch_artist=True, boxprops=dict(facecolor='lightgreen'))
                 plt.title('Content Length by Label')
                 plt.suptitle('')
                 plt.ylabel('Content Length')
                 plt.show()

        else:
            print("Cannot perform distribution analysis or box plot: 'Content_Length' column not found.")

    else:
        print("No numerical columns found for descriptive statistics.")

    if 'Label' in df.columns and 'Source' in df.columns:
        contingency_table = pd.crosstab(df['Label'], df['Source'])
        print("\nContingency Table (Label vs. Source):\n", contingency_table)
    else:
        print("\nCannot create Contingency Table: 'Label' or 'Source' column(s) not found.")
    print("\n")

    # --- Additional Analysis from original notebook ---
    print("--- Additional Analysis ---")

    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        print("Label Distribution:")
        print(label_counts)
        print("\n")

        if 'Source' in df.columns:
            fake_sources = df[df['Label'] == 'FAKE']['Source'].value_counts()
            print("Fake News Sources:")
            print(fake_sources)
            print("\n")

            real_sources = df[df['Label'] == 'REAL']['Source'].value_counts()
            print("Real News Sources:")
            print(real_sources.head(10))
            print("\n")
        else:
             print("Cannot analyze fake/real sources: 'Source' column not found.")
    else:
        print("Cannot analyze label distribution or fake/real sources: 'Label' column not found.")


    if 'Source' in df.columns:
        print("Top Sources:")
        top_sources = df['Source'].value_counts().head(10)
        print(top_sources)
        print("\n")
    else:
        print("Cannot analyze top sources: 'Source' column not found.")

    if 'Language' in df.columns:
        print("Language Distribution:")
        print(df['Language'].value_counts())
        print("\n")
    else:
        print("Cannot analyze language distribution: 'Language' column not found.")


    if 'Date' in df.columns and not df['Date'].isnull().all():
        print("Date Range:")
        print(f"Earliest date: {df['Date'].min()}")
        print(f"Latest date: {df['Date'].max()}")
        print("\n")
    else:
        print("Cannot analyze date range: 'Date' column not found or contains only NaT values.")

    if 'Content' in df.columns and 'Label' in df.columns:
         print("Average Word Count:")
         df['Word_Count'] = df['Content'].astype(str).apply(lambda x: len(x.split()))
         print(f"All articles: {df['Word_Count'].mean():.1f}")
         if 'REAL' in df['Label'].unique():
             print(f"Real articles: {df[df['Label'] == 'REAL']['Word_Count'].mean():.1f}")
         if 'FAKE' in df['Label'].unique():
            print(f"Fake articles: {df[df['Label'] == 'FAKE']['Word_Count'].mean():.1f}")
         print("\n")
    else:
        print("Cannot analyze word count: 'Content' or 'Label' column not found.")

    # --- Visualization (Combined) ---
    print("--- Visualization ---")
    plt.figure(figsize=(15, 10))

    if 'Label' in df.columns:
        plt.subplot(2, 2, 1)
        label_counts.plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('Label Distribution (REAL vs FAKE)')
    else:
        print("Cannot generate Label Distribution pie chart: 'Label' column not found.")


    if 'Source' in df.columns:
        plt.subplot(2, 2, 2)
        top_sources.plot.bar(color='skyblue')
        plt.title('Top 10 Sources')
        plt.xticks(rotation=45)
    else:
        print("Cannot generate Top Sources bar chart: 'Source' column not found.")


    if 'Word_Count' in df.columns and 'Label' in df.columns:
        plt.subplot(2, 2, 3)
        df.boxplot(column='Word_Count', by='Label', grid=False,
                   boxprops=dict(color='black'),
                   medianprops=dict(color='red'))
        plt.title('Word Count by Label')
        plt.suptitle('')
    else:
        print("Cannot generate Word Count box plot: 'Word_Count' or 'Label' column not found.")


    if 'Source' in df.columns and 'Label' in df.columns:
        plt.subplot(2, 2, 4)
        fake_sources.plot.bar(color='salmon')
        plt.title('Fake News Sources')
        plt.xticks(rotation=45)
    else:
        print("Cannot generate Fake News Sources bar chart: 'Source' or 'Label' column not found.")


    plt.tight_layout()
    plt.show()

    # --- Save Analyzed Data ---
    print("--- Saving Analyzed Data ---")
    output_file = 'analyzed_news_data.xlsx'
    try:
        with pd.ExcelWriter(output_file) as writer:
            df.to_excel(writer, sheet_name='All Data', index=False)
            if 'Label' in df.columns:
                label_counts.to_frame().to_excel(writer, sheet_name='Label Counts')
            if 'Source' in df.columns:
                 top_sources.to_frame().to_excel(writer, sheet_name='Top Sources')

        print(f"Analysis complete. Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving analyzed data: {e}")
