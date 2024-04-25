import json
import torch
from data_loader import load_data
from dim_clustering import cluster_texts
from train__nn import split_labeled_embeddings, train_network, evaluate_network

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    texts = load_data(config['data']['dataset'])
    embeddings, topics, topic_model = cluster_texts(texts, config)

  

    # Saving topics information
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(f"topic_info_{config['data']['dataset']}.csv")

    # Saving embeddings and topics
    filename = f"embeddings_topics_{config['data']['dataset']}_umapdim_{config['pca_umap']['umap_n_components']}_hbbscan_eps_{config['hdbscan']['cluster_selection_epsilon']}.pt"
    torch.save({'embeddings': embeddings, 'topics': topics}, filename)



    X_train, X_val, X_test, y_train, y_val, y_test = split_labeled_embeddings(embeddings.numpy(), topics.numpy())
    num_classes = max(topics.unique()) + 2
    print(f'Number of our classes:{num_classes}')


    model = train_network(X_train, y_train, X_val, y_val, num_classes, config)
    evaluate_network(model, X_test, y_test)

   