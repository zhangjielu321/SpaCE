# Introduction
This respository contains the code and dataset for: 
[**SpaCE: a spatial counterfactual explainable deep learning model for predicting out-of-hospital cardiac arrest survival outcome**]

**!!! Data used in repository are mocked data considering the data protection policy**

This paper developed a Spatial Counterfactual Explanation Method, two models are developed in this method: 
one is Spatially Explicit Outcome Prediction (SEP) model, another is Prototype-guided Counterfactual Explanation Model(PCE).
The SEP model is used for fusing the health and spatial features effectively and improve prediction performance.
The PCE model is used for explaining SEP model by selecting prototypes and generating counterfactual examples.

# Installing dependencies
1. Following requirement.txt to install all packages needed
2. pip install geoclip
   https://github.com/VicenteVivan/geo-clip
3. The MMD-Critic folder has been included in our repository
   https://github.com/maxidl/MMD-critic


# Try these three demos 
### Explore SEP model, Prototype Selection(PCE), and Counterfactual Generation(PCE).
Considering the real data is protected, we generate mock data for testing these three demos. These mock data has the same data type and feature type
as our real-scenario data used in our paper.

1. Demo try for SEP model
cd ./v2_analysis_indivilevel/Demo_CounterfactualGeneration_PCE.ipynb
2. Demo try for Prototype Selection(1st part of PCE model)
cd ./v2_analysis_indivilevel/Demo_prototpyeGenerate.ipynb
3. Demo try for Counterfactual Generation(2nd part of PCE model)
cd ./v2_analysis_indivilevel/Demo_CounterfactualGeneration_PCE.ipynb


# Structure for SpaCE repository
```
.
├── readme.md
├── requirements.txt  # Install dependencies
├── v2_analysis_indivilevel  # Experiment
│   ├── Demo_CounterfactualGeneration_PCE.ipynb  # Demo for counterfactual generation
│   ├── Demo_prototpyeGenerate.ipynb  # Demo for prototype generation
│   ├── Demo_VAE_geoclip_SEP.ipynb    # Demo for Spatially Explicit Prediction(SEP) model
│   ├── mock_data                     # Create mock data 
│   │   ├── mock_data_all.csv        
│   │   ├── mock_data_concat.csv
│   │   ├── mock_data_outcome_0.csv
│   │   └── mock_prototype.csv
│   ├── v2_data_generate_peusdo_data.py   # Create simulated data (as shown in paper)
│   ├── v2_main_Counterfactual_total.py   # Get all counterfactuals and global importance
│   ├── v2_main_Couterfactual_Region.py   # Compute county-level importance
│   ├── v2_model_Base.ipynb               # Run baseline models in comparison with SEP
│   ├── v2_model_VAE_geoclip_visualization.ipynb # Visualize embedding learned from SEP model 
│   ├── v2_plot_GlobalImportance.ipynb    # Plot global importance and coefficient
│   ├── v2_plot_LocalImportance.ipynb     # Plot individual importance and coefficient
│   ├── v2_plot_Region_Importance.ipynb   # Plot county-level importance and coefficient
│   └── v2_repackaged_forDice_model.pth    
├── v2_dice.py           # Run baseline DiCE model in comparison with PCE model
├── v2_diceVSourmethod.ipynb # Calculate Evaluation Metric
├── v2_MMD_critic   # Original MMD_critic model for prototype generation
│   ├── digits.py
│   ├── imagenet.py
│   ├── __init__.py
│   ├── kernels.py
│   ├── mmd_critic.py
│   └── README.md
└── v2_utils     # Core .py files for our SpaCE method
    ├── feature_importance.py  # class for importance calculation
    ├── generate_counterfactual.py  # class for Counterfactual Generation
    ├── generate_prototype.py   # class for Prototype Generation
    ├── __init__py
    ├── visualize_dataframe.py # class for visualize dataframe

```
