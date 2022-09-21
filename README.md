Repository overview:
-


Install requirements:
  -

- Installations required for reproduction (can all be installed with pip):
  - Requirements:
    - pandas
    - matplotlib
    - tensorflow
    - keras
    - active-semi-supervised-learning (note that example_oracle.py in this library needs to be replaced with the example_oracle.py in this repo for the code to work)
    - seaborn
    - scikit-learn
    - colorcet
    

Data Overview:
  -

- Data:
  - full_dataset.csv -- download from https://drive.google.com/file/d/11HZ0I3jYplkk6u6o4JxQWhnBRoSZJLGN/view?usp=sharing
  - final_dataset.csv -- slimmed down version of master file "full_dataset.csv", which was too large to publish here. Contains all binary featurs for keyword commands, as well as session duration and total amount of commands used
  - expert_dataset.csv -- file containing all input. Used for creation of the sets containing the clustering results, as well as the label set
  - results_KMeans.csv -- results of the KMeans clustering algorithm
  - results_PCKmeans_exp.csv -- results of the PCKMeans clustering algorithm
  - results_SeededKmeans.csv -- results of the SeededKMeans algorithm
  - cluster_labels.csv -- contains all labelled sessions for supervision purposes
  - combined_results.csv -- contains the combined results of all clustering algorithms 
  - encoder_C402040.h5 -- encoder model for feature condensing
  - example_oracle.py -- oracle used in the PCKMeans model. Original in package needs to be replaced with this for the PCKMeans model to work.






Reproduction instructions:
  -


Instructions for reproducing the results in this thesis:

1. Run create_features.py -- creates final_dataset.csv and expert_dataset.csv
2. Run explore_data.py -- creates plots and removes outliers
3. Run feature_construction.py -- Creates condensed feature set. Note that it is adviced to skip this step for reproduction and just use the encoder_C402040.h5 model
4. Run the clustering models:
   1. PCKmeans_expert.py -- creates results_PCKmeans_exp.csv and calculates AMI for model configs
   2. kmeans.py -- creates results_kMeans.csv and calculates AMI for model configs
   3. SeededKmeans.py -- creates results_SeededKmeans.csv and calculates AMI for model configs
5. Run combine_results.py -- Combines all results into combined_results.csv and creates different plots
