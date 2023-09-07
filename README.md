# Information Flow-Based Fuzzy Cognitive Maps with Enhanced Interpretability
[![View Private Cody Leaderboard on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/products/matlab.html) [![Generic badge](https://img.shields.io/badge/Python-Powered-<COLOR>.svg)](https://www.python.org/)


## Table of Contents
1. [General Info](#general-info)
2. [Abstract](#abstract)
3. [Dataset](#dataset)
4. [Scripts](#scripts)
5. [Citation](#citation)

### General Info
***
This repository contains the source code developed and used in the paper: *"Information Flow-Based Fuzzy Cognitive Maps with Enhanced Interpretability"* written by **Marios Tyrovolas**, **X. San Liang**, and **Chrysostomos Stylios**. 
***

<p align="center">
<img src="proposed_methodology_block%20diagram.png" alt="Proposed Methodology" width="400">
</p>

### Abstract
***
Fuzzy Cognitive Maps (FCMs) are a graph-based methodology successfully applied for knowledge representation of complex systems modelled through an interactive structure of nodes connected with causal relationships. Due to their flexibility and inherent interpretability, FCMs have been used in various modelling and prediction tasks to support human decisions. However, a notable limitation of FCMs is their susceptibility to inadvertently capturing spurious correlations from data, undermining their prediction accuracy and interpretability. In addressing this challenge, our primary contribution is the introduction of a novel framework for constructing FCMs using the Liang-Kleeman Information Flow (L-K IF) analysis, a quantitative causality analysis rigorously derived from first principles. The novelty of the proposed approach lies in the identification of actual causal relationships from the data using an automatic causal search algorithm. These relationships are subsequently imposed as constraints in the FCM learning procedure to rule out spurious correlations and improve the aggregate predictive and explanatory power of the model. Numerical simulations validate the superiority of our method against state-of-the-art FCM-based models, thereby bolstering the reliability, accuracy, and interpretability of FCMs.
***

## Dataset

We adopted Matzka’s PMAI4I dataset to perform the experiments, a synthetic yet realistic dataset representing industrial predictive maintenance data.

[S. Matzka, “Explainable artificial intelligence for predictive maintenance applications,” in *2020 Third International Conference on Artificial Intelligence for Industries (AI4I).* IEEE, Sep. 2020](https://ieeexplore.ieee.org/document/9253083)

The provided [link](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) contains the original repository of the dataset. To access the dataset, including the folds resulting from the stratified k-fold cross-validation, please navigate to the [*"dataset"*](https://github.com/marios-tyrovolas/A-Novel-Framework-for-Enhanced-Interpretability-in-Fuzzy-Cognitive-Maps/tree/main/dataset) folder.

## Scripts

A list of scripts used within the paper:

II. THEORETICAL BACKGROUND

* [**Experiment for L-K IF Analysis on Binary Time Series**](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/LK_IF_anal_binary_data_expr): 
  1. *LK_IF_anal_binary_data_expr*

III. PROPOSED METHODOLOGY

* [**Data pre-processing**](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/data_preprocessing): 
  1. *AI4I2020_data_preprocessing.ipynb*
  2. *GL_scale_normalization* 
* [**Training Phase 1 (L-K IF Analysis)**](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/IF_based_causality_analysis_ai4i2020):
  1. *IF_based_causality_analysis_ai4i2020* 
  2. *multi_causal_lyap_est.m*
  3. *multi_causality_est.m*
  4. *multi_causality_est_all_new.m*
  5. *multi_tau_est.m*

IV. [COMPARATIVE ANALYSIS AGAINST FCM-BASED AND ML MODELS](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/experimental_setup)
  
   * [Machine Learning Models (ML)](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/experimental_setup/ML)
     1. *Training and Evaluating other ML models_AI4I_2020_Final_version.ipynb*
     2. *Global_Feat_Import_ML_mdl*
  * FCM-based Models
    * [IFFCM](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/experimental_setup/IFFCM)
      1. IF-FCM Learning Algorithm
      3. Predictive Power Evaluation
      4. Explatory Power Evaluation
    * [FCMB and FCMMC](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/experimental_setup/FCMB_FCMMC)
      1. *Classification and feature transformation with Fuzzy Cognitive Maps - Piotr Szwed.ipynb*
      2. *Hyper-parameter Tuning in FCMB and FCMMC.ipynb*
    * [LTCN](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/experimental_setup/LTCN)
      1. *Long-Term Cognitive Network for Pattern Classification.ipynb*
    * [FCM-FC](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/experimental_setup/FCM_FC)
      1. *Local_Interpretabiity_FCM_FC.m*
      2. *Particle_Swarm_Optimization_FCM_FC.m*
      3. *fcm_fc_training_cost_function.m*
    * [CCFCM](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/experimental_setup/CCFCM)
      1. *Correlation_coefficients_ai4i2020.m*
      2. *Local_Interpretability_Corr_Coef_FCM.m*
    * [FCM-SSF](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/experimental_setup/FCM_Expert_approach)
      1. *FCM_Expert_approach*
    * [FCM-A](https://github.com/marios-tyrovolas/Information-Flow-Based-Fuzzy-Cognitive-Maps-with-Enhanced-Interpretability/tree/main/experimental_setup/Froelich_approach)
      1. *Froelich_approach*

## Citation

If you find our code useful, please cite our paper. 

```
@article{tyrovolas_information_2023,
	title = {Information flow-based fuzzy cognitive maps with enhanced interpretability},
	issn = {2364-4974},
	url = {https://doi.org/10.1007/s41066-023-00417-7},
	doi = {10.1007/s41066-023-00417-7},
	journal = {Granular Computing},
	author = {Tyrovolas, Marios and Liang, X. San and Stylios, Chrysostomos},
	month = sep,
	year = {2023},
}
```

### Contact

For any question, please raise an issue or contact

```
Marios Tyrovolas: tirovolas@kic.uoi.gr
```
### Acknowledgement

Open access funding provided by HEAL-Link Greece. We acknowledge the support of this work by the project "Dioni: Computing Infrastructure for Big-Data Processing and Analysis." (MIS No. 5047222) which is implemented under the Action "Reinforcement of the Research and Innovation Infrastructure", funded by the Operational Programme "Competitiveness, Entrepreneurship and Innovation" (NSRF 2014–2020) and cofinanced by Greece and the European Union (European Regional Development Fund). XSL is partially funded by the National Science Foundation of China under Grant #42230105, and by Southern Marine Science and Engineering Guangdong Laboratory (Zhuhai) through the startup foundation and scientific research program.

We also extend our gratitude to Prof. Piotr Szwed for generously sharing the source code of [FCMB/FCMMC](https://github.com/pszwed-ai/fcm_classifier_transformer) and to Prof. Gonzalo Nápoles for providing the source code of [LTCN](https://github.com/gnapoles/ltcn-classifier). Their contributions were instrumental in conducting the comparative study presented in this manuscript.

