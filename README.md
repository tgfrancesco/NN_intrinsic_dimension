# The Intrinsic Dimension of Neural Networks Ensembles

This repository contains the code and data used for the study:

**"The Intrinsic Dimension of Neural Networks Ensembles"**, Tosti Guerra, F.; Napoletano, A.; Zaccaria

## Requirements

To run the code, you need the following dependencies:

- Python 3.x
- Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

To use MLE estimator -> https://github.com/stat-ml/GeoMLE.git
https://www.mathworks.com/matlabcentral/fileexchange/40112-intrinsic-dimensionality-estimation-techniques contains the implementation of MiND_ML, MiND_KL, DANCo, DANCoFit.

## Workflow

- The notebook **NN_generator** contains scripts to generate N neural networks to build the ensembles described in the paper.
- Once the ensemble is generated, the notebook **NN_ID_calculator** can be used to evaluate the Intrinsic Dimension of the produced ensemble.
- The notebook **Plots** contains scripts to generate the plots included in the paper.
