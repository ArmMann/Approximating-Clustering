import pandas as pd
from datasets import load_dataset
from abc import ABC, abstractmethod

class DataLoader(ABC):
    @abstractmethod
    def load_data(self, dataset_name):
        pass

class DefaultDataLoader(DataLoader):
    def load_data(self, dataset_name):
        dataset = load_dataset(dataset_name)
        df = dataset['train'].to_pandas()
        return df['text'].tolist()

class TweetEvalDataLoader(DataLoader):
    def load_data(self, dataset_name):
        dataset = load_dataset(dataset_name, 'emoji')
        dataframes = {split: dataset[split].to_pandas() for split in dataset.keys()}
        all_data = pd.concat(dataframes.values(), ignore_index=True)
        return all_data['text'].tolist()

def get_loader(dataset_name):
    if dataset_name == "tweet_eval":
        return TweetEvalDataLoader()
    else:
        return DefaultDataLoader()