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

### Abstract
***
Fuzzy Cognitive Maps (FCM) is a graph-based methodology successfully applied for knowledge representation of complex systems modelled through an interactive structure of nodes connected with causal relationships. Due to their flexibility and inherent interpretability, FCMs have been used in various modelling and prediction tasks to support human decisions. However, one of the main limitations of FCMs is that they may unintentionally absorb spurious correlations from the collected data, resulting in poor prediction accuracy and interpretability. This article proposes a novel framework for constructing FCMs based on Liang-Kleeman Information Flow (L-K IF) analysis to address this limitation. The novelty of the proposed approach is the identification of actual causal relationships from the data using an automatic causal search algorithm.  The actual causal relationships are then imposed as constraints in the FCM learning procedure to rule out spurious correlations and improve the predictive and explanatory power of the model. Numerical simulations were conducted to demonstrate the effectiveness of the proposed approach by comparing it with state-of-the-art FCM-based models. The code for this study is available in this repository.
***

## Dataset

We adopted Matzka’s PMAI4I dataset to perform the experiments, a synthetic yet realistic dataset representing industrial predictive maintenance data.

[S. Matzka, “Explainable artificial intelligence for predictive maintenance applications,” in *2020 Third International Conference on Artificial Intelligence for Industries (AI4I).* IEEE, Sep. 2020](https://ieeexplore.ieee.org/document/9253083)

The provided [link](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) contains the original repository of the dataset. To access the dataset, including the folds resulting from the stratified k-fold cross-validation, please navigate to the [*"dataset"*](https://github.com/marios-tyrovolas/A-Novel-Framework-for-Enhanced-Interpretability-in-Fuzzy-Cognitive-Maps/tree/main/dataset) folder.

## Scripts

A list of scripts used within the paper:

II. THEORETICAL BACKGROUND

* [**Experiment for L-K IF Analysis on Binary Time Series**]((https://github.com/marios-tyrovolas/A-Novel-Framework-for-Enhanced-Interpretability-in-Fuzzy-Cognitive-Maps/tree/main/LK_IF_anal_binary_data_expr)): 
  1. *LK_IF_anal_binary_data_expr*

III. PROPOSED METHODOLOGY

* **Data pre-processing**: 
  1. *AI4I2020_data_preprocessing.ipynb*
  2. *k-fold cross validation datasets*
  3. *GL_scale_normalization* 
* **Training Phase 1 (L-K IF Analysis)**:
  1. *IF_based_causality_analysis_ai4i2020* 
  2. *taus.csv*
* **Training Phase 2 (IF-FCM Learning Algorithm)**
  1. [Global Optimization Toolbox](https://www.mathworks.com/products/global-optimization.html): Version R2021b
  2. **PSO Algorithm**: *Particle_Swarm_Optimization_Napoles_Error_Function.m*
  3. **Cost Function**: *Napoles_improved_error_function*
  4. **Near-optimal solution for each fold**: *near_opt_sol_for_folds*
* **What-if Simulations, Threshold moving, and evaluation metrics for IF-FCM's predictive power**
  1. *Constructed_FCM_Simulations_for_Napoles_error_function.m*
* **Global and Local Interpretability for IF-FCM**
  1. *Global_Interpretability.m*
  2. *Local_Interprtability.m*

IV. COMPARATIVE ANALYSIS AGAINST OTHER ML AND FCM-BASED MODELS
  
   * Machine Learning Models
     1. *Training and Evaluating other ML models_AI4I_2020_Final_version.ipynb*
     2. *Global_Feat_Import_ML_mdl*
  * FCM-based Models      
    * [FCMB and FCMMC](https://github.com/pszwed-ai/fcm_classifier_transformer)
      1. *Classification and feature transformation with Fuzzy Cognitive Maps - Piotr Szwed.ipynb*
      2. *Hyper-parameter Tuning in FCMB and FCMMC.ipynb*
    * [LTCN](https://github.com/gnapoles/ltcn-classifier)
      1. *Long-Term Cognitive Network for Pattern Classification.ipynb*
    * [FCN-FW](https://www.sciencedirect.com/science/article/pii/S1568494621003380)
      1. *blah_blah.m*
    * [FCM-SSF](https://sites.google.com/view/fcm-expert?pli=1)
      1. *FCM_Expert_approach*
    * [FCM-A](https://www.sciencedirect.com/science/article/pii/S0925231216315703)
      1. *Froelich_approach*


PS: A second possible solution for FCM training was also examined during this research. Specifically, the normalized IFs were used as the FCM weights but were subject to tuning according to their confidence intervals. For the completeness of the scripts, this code is also listed.
1. *Particle_Swarm_Optimization_Option_2.m*
2. *confidence_intervals_taus.mat*

## Citation

If you find our code useful, please cite our paper. 

```
@article{Tyrovolas2023,
author = "Marios Tyrovolas and X. San Liang and Chrysostomos Stylios",
title = "{A Novel Framework for Enhanced Interpretability in Fuzzy Cognitive Maps}",
year = "2023",
month = "5",
url = "https://www.techrxiv.org/articles/preprint/A_Novel_Framework_for_Enhanced_Interpretability_in_Fuzzy_Cognitive_Maps/22718032",
note = {TechRxiv techrxiv.22718032},
doi = "10.36227/techrxiv.22718032.v1"
}
```

### Contact

For any question, please raise an issue or contact

```
Marios Tyrovolas: tirovolas@kic.uoi.gr
```
### Acknowledgement

We acknowledge the support of this work by the project *"Dioni: Computing Infrastructure for Big-Data Processing and Analysis."* (**MIS No. 5047222**) which is implemented under the Action *"Reinforcement of the Research and Innovation Infrastructure"*, funded by the Operational Programme *"Competitiveness, Entrepreneurship and Innovation"* (**NSRF 2014-2020**) and co-financed by Greece and the European Union (European Regional Development Fund).

We also extend our gratitude to Prof. Piotr Szwed for generously sharing the source code of [FCMB/FCMMC](https://github.com/pszwed-ai/fcm_classifier_transformer) and to Prof. Nápoles for providing the source code of [LTCN](https://github.com/gnapoles/ltcn-classifier). Their contributions were instrumental in conducting the comparative study presented in this manuscript.

