from sklearn.decomposition import PCA
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PCA_UMAP:
    def __init__(self, config):
        self.pca = PCA(n_components=config['pca_components'])
        self.umap = UMAP(n_neighbors=config['umap_n_neighbors'],
                         n_components=config['umap_n_components'],
                         min_dist=config['umap_min_dist'],
                         metric=config['umap_metric'],
                         random_state=42)

    def fit(self, X, y=None):
        self.pca.fit(X)
        X_reduced = self.pca.transform(X)
        self.umap.fit(X_reduced, y=y)
        return self

    def transform(self, X, y=None):
        X_reduced = self.pca.transform(X)
        return self.umap.transform(X_reduced)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

def cluster_texts(texts, config):
    sentence_model = SentenceTransformer(config['bertopic']['embedding_model'])
    embeddings = sentence_model.encode(texts, show_progress_bar=True)
    
    dim_model = PCA_UMAP(config['pca_umap'])
    #umap_embeddings = dim_model.fit_transform(embeddings)

    hdbscan_model = HDBSCAN(min_cluster_size=config['hdbscan']['min_cluster_size'],
                            min_samples=config['hdbscan']['min_samples'],
                            metric=config['hdbscan']['metric'],
                            cluster_selection_method=config['hdbscan']['cluster_selection_method'],
                            cluster_selection_epsilon=config['hdbscan']['cluster_selection_epsilon'],
                            prediction_data=True)

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=config['bertopic']['reduce_frequent_words'])
    
    topic_model = BERTopic(embedding_model=sentence_model,
                           umap_model=dim_model,
                           ctfidf_model=ctfidf_model,
                           hdbscan_model=hdbscan_model)
    
    print('Starting Initial Stack Fitting')
    topics, probs = topic_model.fit_transform(texts)
    return torch.tensor(embeddings), torch.tensor(topics), topic_model

