# GLocalKD
This is the code for WSDM2022 paper "Deep Graph-level Anomaly Detection by Glocal Knowledge Distillation".

## Brief Introduction
Global and Local Knowledge Distillation (GLocalKD) is introduced in our WSDM22 paper, which leverages a set of normal training data to perform end-to-end anomaly score learning for graph-level anomaly detection (GAD) problem. GAD describes the problem of detecting graphs that are abnormal in their structure and/or the features of their nodes, as compared to other graphs. GLocalKD addresses a semi-supervised GAD problem in that the data known are all labeled normal data. The experiment results show that  GLocalKD can be implemented data-effectively and is robustness to anomaly contamination, indicating its applicability in both unsupervised (anomaly-contaminated unlabeled training data) and semi-supervised (exclusively normal training data) settings.

## Data Preparation

Some of datasets are put in ./dataset folder. Due to the large file size limitation, some datasets are not uploaded in this project. You may download them from the urls listed in the paper.

## Large Size Dataset
```shell
aws s3 sync s3://prod-xxxx-knowledge-lake-sandpit-v1/tmp/data/glocalkd/streamspot/ dataset/streamspot/
```
## Train

For datasets except HSE, p53, MMP, PPAR-gamma and hERG, run the following code. For datasets with node attributes, feature chooses default, otherwise deg-num.
```shell
python main.py --DS [] --feature [default/deg-num]
```

For HSE, p53, MMP and PPAR-gamma, run the following code.
```shell
python main_Tox.py --DS []
```

For hERG, run the following code.
```shell
python main_smiles.py
```

```
python src/main.py --DS streamspot --batch-size 1 --hidden-dim 32 --output-dim 32 --num_epochs 51 --fix-train-test True --lr 0.01 --num-node-types 8 #--test False
python src/main.py --DS TraLog --batch-size 128 --hidden-dim 128 --output-dim 128 --num_epochs 51 --fix-train-test True --lr 0.0001 --num-node-types 8 #--test False
```


## Citation
```bibtex
@inproceedings{ma2022deep,
  title={Deep Graph-level Anomaly Detection by Glocal Knowledge Distillation},
  author={Ma, Rongrong and Pang, Guansong and Chen, Ling and van den Hengel, Anton},
  booktitle={WSDM '22: The Fifteenth ACM International Conference on Web Search and Data Mining},
  year={2022}
}
```
