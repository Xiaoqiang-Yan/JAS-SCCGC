# Self-cumulative Contrastive Graph Clustering
We provide a demo based on ACM, DBLP and Pubmed datasets for our submitted paper: [Self-cumulative Contrastive Graph Clustering](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2024.125025).  Run main.py to show the training process of the model.

   


## Requirements
python==3.9
pytorch==1.12.0
numpy==1.23.5
munkres==1.1.4

## Datasets

The ACM, DBLP and Pubmed datasets are placed in "data" folder. The others dataset can find on the Internet. We will not put the link here

## Usage

The pre-train: [ae_pretrain.py](../ae_pretrain/ae_pretrain.py)
Before model training, we pretrain the auto-encoder for 50 epochs with learning rate 0.001 to generate the initialized feature representations of the benchmark datasets. Then, k-means algorithm is conducted 20 times to obtain the initialized cluster centroid based on the learned feature representations from the pre-trained auto-encoder. 

The model train: [main.py](main.py)
```train
python main.py 
```


## 5.Experiment Results
As shown in the paper: [Self-cumulative Contrastive Graph Clustering](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2024.125025).
