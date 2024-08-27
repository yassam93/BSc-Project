# BSc-Project Repository
 
## Overview
This repository has the entirety of the datasets and source code employed for my BSc Data Science and Computing project, which focusses on scrutinising the influence of lifestyle factors on sleep disorders through the utilisation of machine learning methodologies. The following is a detailed analysis of the contents of the repository:

## Repository Contents
 
### Folders:
- **datasets/**:This folder contains three essential datasets utilised in the analyses:
  - `Sleep_health_and_lifestyle_dataset.csv` is the initial dataset obtained from Kaggle. It includes demographic, behavioural, and physiological factors.
  - `preprocessed_dataset.csv` file contains the dataset that has undergone initial preprocessing.
  - `preprocessed_dataset_scaled_columns.csv` is a dataset that contains scaled features for "Age", "Heart Rate", "Daily Steps", "Physical Activity Level". Scaling the features ensures that each variable contributes equally, which improves the accuracy and dependability of the model.

### Code Files:
- `preprocess.py` is a Python script that performs data preparation tasks such as data cleansing, feature engineering, and data scaling.
- `logistic-regression.R` is a R script that implements the logistic regression model. It is used to analyse the correlation between lifestyle factors and sleep disorders.
- `SVM.R` is a R script that implements Support Vector Machine models and explores various kernels to enhance the accuracy of predictions.

## Instructions for Use
1. Clone this repository on your local system 
2. Verify that you have Python and R installed on your PC.
3. Execute the `preprocess.py` script to carry out data preprocessing.
4. Utilise the `logistic-regression.R` and `SVM.R` scripts to perform the analysis and produce model outputs.

## Supplementary Details 
- Each script is thoroughly documented with comments elucidating the actions and approaches employed.
The user did not provide any text. 
- The datasets located in the `datasets/` folder are utilised in various scripts for the purpose of analysis and training of machine learning models.

## Supplementary Materials
To access comprehensive citations and thorough analysis of the methodology and data sources employed in this study, please consult the entire project report. The report contains extensive citations and a more extensive analysis of the research findings within a wider perspective.
