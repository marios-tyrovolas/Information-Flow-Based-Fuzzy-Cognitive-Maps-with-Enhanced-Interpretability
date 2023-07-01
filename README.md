# A Novel Framework for Enhanced Interpretability in Fuzzy Cognitive Maps
[![View Private Cody Leaderboard on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/products/matlab.html)

[![Generic badge](https://img.shields.io/badge/Python-Powered-<COLOR>.svg)](https://www.python.org/)


## Table of Contents
1. [General Info](#general-info)
2. [Dataset](#dataset)
3. [Scripts](#scripts)
4. [Reference](#reference)

### General Info
***
This repository contains the source code developed and used in the paper: *"A Novel Framework for Enhanced Interpretability in Fuzzy Cognitive Maps"* written by **Marios Tyrovolas**, **X. San Liang**, and **Chrysostomos Stylios**. 
***

## Dataset

We adopted Matzka’s PMAI4I dataset to perform the experiments, a synthetic yet realistic dataset representing industrial predictive maintenance data.

[S. Matzka, “Explainable artificial intelligence for predictive maintenance applications,” in *2020 Third International Conference on Artificial Intelligence for Industries (AI4I).* IEEE, Sep. 2020](https://ieeexplore.ieee.org/document/9253083)
 

## Scripts

A list of scripts used within the paper:

II. THEORETICAL BACKGROUND

* **Experiment for L-K IF Analysis on Binary Time Series**: 
  1. *IF-based analysis Liang's experiment*

III. PROPOSED METHODOLOGY

* **Data pre-processing**: 
  1. *Fuzzy Cognitive Maps in Classification AI4I Dataset.ipynb*
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

## Reference

Tyrovolas, M., San Liang, X., & Stylios, C. (2023). A Novel Framework for Enhanced Interpretability in Fuzzy Cognitive Maps.
