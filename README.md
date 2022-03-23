# GLIM
This repository contains code for the paper "Decoding multi-level relationships with human tissue-cell-molecule network".

## Hardware requirements
GLIM can run on CPU-only hardware, but GPU with at least 6GB of VRAM is recommended.

## Software requirements
GLIM has been tested on Linux. GLIM requires Python 3.6. All other major dependencies include torch 1.6.0, torch-geometric 1.6.1, numpy 1.19.4 and so on.

## Installing
To download GLIM to your home directory, run
```
git clone https://github.com/Svvord/GLIM.git
```

We also recommend using conda to run GLIM.
```
conda create -n GLIM python=3.6
conda activate GLIM
```
These step should take a few minutes.

## Getting started

The HMLN relationship data is available in `/data/relationship_table.txt`.
The public graph datasets used in paper are available through following link:

https://cloud.tsinghua.edu.cn/f/ae74c356b31e436f881f/?dl=1

Two extra tutorials are available in `/tutorials/`

## Start runing

To accelerate the running process, a preprocessed node feature file can be downloaded through:

https://cloud.tsinghua.edu.cn/f/ae74c356b31e436f881f/?dl=1

Of course, our program also supports ab initio calculations.

### Configuring and starting the run

```
python3 glim_embedding.py \
--relationship-file "./data/relationship_table.txt" \
--node-feature-file "./data/node_feature.npy" \
--embedding-save-file "./results/hmln_feature.npy"
```

`relationship-file` contains at least two columns, which represents two nodes of the pairs.

`node-feature-file` contains original node's feature, Note that the order of the nodes should be sorted in the order of their names! Otherwise, you will get wrong embedding results. For details, you can refer to the `node_map` construction method.

```
node_map = {item:i for i, item in enumerate(sorted(list(set(node_list))))}
```

`embedding-save-file` contains the embedding vectors of GLIM.

![Fig1-Overview](/Fig1-Overview.png)
