![logo](./assets/logo.png)
## LLM2BinFuncSim: Hybrid LLM + GNN Model for Binary Function Similarity

[![Python 3.11](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3109/)  [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

### About

This repository implements a hybrid approach using Large Language Models (LLMs) and Graph Neural Networks (GNNs) to tackle the binary function similarity problem in cybersecurity.

### Step 0: Preparations
Run the following command to prepare the development environment

```bash
poetry install --no-root && bash bootstrap.sh
```

### Step 1: Data Preparation

The first step involves generating a graph dataset from existing cybersecurity function datasets used in previous works. You'll find sample code for this step in `generate_graph_dataset.py`.

To generate the graph dataset, execute the following commands:

```bash
python preprocess/generate_graph_dataset.py
python preprocess/filter_graph_dataset.py
```

### Step 2: Training

The training step can be divided in two phases LLM pretraining and then also the GNN training.

#### Step 2a LLM training
For the first one we can choose to first domain adapt (`da`) the LLM or pretrain the LLM in a supervised contrastive training fashion (`sct`) running the example scripts presents in the `examples` folder. 

Make sure to set the correct file paths for the data processed in Step 1.
To extract the CLS from the pretrained model look at the `examples/emb_extraction.sh` script.

#### Step 2b GNN training
In this part you have to provide the node embeddings extracted in the previous step for each dataset split to the GNN, in order to achieve this run the following commands in the project root directory:

```bash
cd ./HermesSim
# Modify preprocess_all.sh script as needed to process configuration JSON files
bash preprocess/preprocess_all.sh
# Train the GNN model
python model/main.py --inputdir dbs --config ./model/configures/llm.json --dataset=one
```  
Ensure the configuration JSON files (for training, validation, and testing) are correctly formatted. Example JSON files are provided in the `preprocess` folder.

### Step 3: Evaluation
For evaluating the model, please refer to the evaluation protocol described in the [HermesSim repository](https://github.com/NSSL-SJTU/HermesSim) . This will guide you on how to evaluate the similarity results between binary functions.