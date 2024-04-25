from datasets import load_dataset
import pandas as pd

# the one t load tweet_eval:
def load_data(dataset_name):
    # Load the dataset with all available splits
    dataset = load_dataset(dataset_name, 'emoji')
    
    # Convert all splits to pandas DataFrames and store in a dictionary
    dataframes = {split: dataset[split].to_pandas() for split in dataset.keys()}
    
    # Concatenate all dataframes into one
    all_data = pd.concat(dataframes.values(), ignore_index=True)
    
    # Return the 'text' column as a list
    return all_data['text'].tolist()

"""def load_data(dataset_name):
    dataset =  load_dataset(dataset_name)

    df = dataset["train"].to_pandas()
    return df['text'].tolist()
"""