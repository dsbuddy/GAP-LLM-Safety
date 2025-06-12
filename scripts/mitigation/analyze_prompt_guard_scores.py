# Read in csv to dataframe
import pandas as pd

def read_csv_to_df(csv_file_path):
    """
    Reads a csv file into a pandas dataframe
    """
    df = pd.read_csv(csv_file_path)
    return df

if __name__ == '__main__':
    csv_file_path = 'prompt_guard_scores_advbench.csv'
    df = read_csv_to_df(csv_file_path)
    print(df.head())

    # Round each of the scores to 0 decimal places and then compute how mnay 0s and 1s there are for each method
    df['Prompt Guard Score'] = df['Prompt Guard Score'].round(0).astype(int)
    print(df.groupby('Method')['Prompt Guard Score'].value_counts().unstack().fillna(0))
