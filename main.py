import pandas as pd

# Read the anime dataset
df_anime = pd.read_csv('csv/anime_cleaned.csv')

# Clean and transform data
def clean_anime_data(df):
    # Convert aired dates to datetime
    df['aired_from_year'] = pd.to_datetime(df['aired_from_year'], errors='coerce')
    
    # Convert duration to numeric minutes
    df['duration_min'] = pd.to_numeric(df['duration_min'], errors='coerce')
    
    # Convert numeric columns
    numeric_columns = ['score', 'scored_by', 'rank', 'popularity', 'members', 'favorites']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean genre strings and convert to list
    df['genre'] = df['genre'].fillna('')
    df['genre'] = df['genre'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    
    # Clean status
    df['status'] = df['status'].fillna('Unknown')
    
    # Convert boolean columns
    df['airing'] = df['airing'].astype(bool)
    
    return df

# Apply cleaning
df_anime_clean = clean_anime_data(df_anime)

# Basic validation
print("Dataset shape:", df_anime_clean.shape)
print("\nMissing values:\n", df_anime_clean.isnull().sum())
print("\nData types:\n", df_anime_clean.dtypes)

