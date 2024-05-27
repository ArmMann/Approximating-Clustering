# Neural Network Based Clustering Method

## Project Overview

Welcome to our GitHub repository! This project is part of a thesis that explores a novel approach to sequential dimensionality reduction and clustering using neural networks. Our primary objective is to accelerate the inference process while maintaining high accuracy.

## Methodology

We've developed a model that dynamically learns from the outcomes of various configurations of the following algorithms:

- **PCA (Principal Component Analysis)** - [Jolliffe, 2016]
- **UMAP (Uniform Manifold Approximation and Projection)** - [McInnes et al., 2018]
- **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)** - [Malzer and Baum, 2019]

These algorithms form a flexible stack that we have tailored for fast and effective learning. Users can easily adjust or swap these components based on their specific needs.
The clusters obtained from the mentioned stack are then used as labels to train a classification network. 

## Application

The focus of our initial model is on textual data, aiming to enhance topic modeling and accelerate data processing times. This setup can be readily adapted to other data domains with some modifications and appropriate preprocessing steps.
Refer to thesis for the results on 3 textual datasets.

## Customization

This model is designed with adaptability in mind, allowing for approximations and clustering on diverse datasets beyond text. Users can experiment with different preprocessing methods and configurations to best suit their unique data challenges.

## Getting Started


### Installation
1. **Clone the Repository**

2. **Install Required Python Packages**
  ```
  pip install -r requirements.txt
  ```

3. **Running the Application**
  ```
  python main.py
  ```

## Documentation
For more detailed information on the configuration and modules, refer to the individual files in the repository:
- `config.json`: Contains configuration settings for the models and data processing.
- `data_loader.py`: Responsible for loading and preprocessing data.
- `dim_clustering.py`: Implements the dimensionality reduction and clustering logic.
- `main.py`: The main script for running the model training and evaluation.

