# Automatic Strain Sensor Design via Active Learning and Data Augmentation for Soft Machines

## Installation ##
The requirement environment is as follow
* python >= 3.6
* matplotlib ~= 3.3.3
* tensorflow ~= 2.1.0
* scikit-learn ~= 0.23.2
* numpy ~= 1.19.2
* pandas ~= 1.1.3
* openpyxl ~= 3.0.5
* deap ~= 1.3.1
* seaborn ~= 0.10.1
* scipy ~= 1.5.2
* six ~= 1.15.0
* imblearn ~= 0.0
* skopt ~= 0.8
* xlrd >= 1.0.0

## Explanation of codes ##
There are the codes for the paper of Automatic Strain Sensor Design via Active Learning and Data Augmentation for Soft Machines

own_package contains most of the useful functions used in this project.

Inside own_package there are following scripts:
* 1.active_learning.py
* 2.analysis.py
* 3.cross_validation.py
* 4.data_agumentation.py
* 5.feature_labels_setup.py
* 6.ga_combination.py
* 7.hparam_opt.py
* 8.inverse_design.py
* 9.models.py
* 10.others.py
* 11.preprocessing.py
* 12.pso_ga.py
* 13.svm_classifier.py

1.active_learning.py contains the main code used in the active learning loop, including the functions to load the trained SVM , select ensembled decision programs, and calculate the acquisition function during each learning loop. It is discussed in paper's Supplementary Information Note S5.

2.analysis.py contains the main code to implement statistical analyses based on Pearson’s coefficient. It is discussed in paper's Supplementary Information Note S8.

3.cross_validation.py contains the main code to realize k-fold cross validation during model training process with different evaluation criteria such as MSE and MRE.

4.data_agumentation.py contains the main code for data augmentation including SMOTE-REG and UIP methods, which is discussed in paper's Supplementary Information Note S7.

5.feature_labels_setup.py contains the main code to construct the recipe labels.

6.ga_combination.py contains the main code for GA to select the best decision programs. It is discussed in paper's Supplementary Information Note S7.

7.hparam_opt.py contains the main code for hyperparameter optimization of all the models emerged in the paper.

8.inverse_design.py contains the main code for running inverse strain sensor design. By entering the design requests to the prediction model, several feasible fabrication recipes were suggested by this optimizer. It is discussed in paper's Supplementary Information Note S9.

9.models.py contains the main code to build the different model classes which could be extracted by different programs inside this project.

10.others.py contains other functions that can be used on demand.

11.preprocessing.py contains the main code for data pre-processing such as loading data from excel, as well as the preliminary data processing to convert recipe and strain labels into the form we want.

12.pso_ga.py contains the main code for particle swarm optimization method.

13.svm_classifier.py contains the main code for constructing SVM classifier. It is discussed in paper's Supplementary Information Note S3.

To repeat the ML tasks inside our paper, one can just utilize all the different run.py inside the folder of own_package. There are:

* 1.run_active_learning.py
* 2.run_cross_validation.py
* 3.run_ga_combination.py
* 4.run_hyparam_opt.py
* 5.run_inverse_design.py
* 6.run_pearson.py
* 7.run_preprocessing.py
* 8.run_svm_classifier.py

1.run_active_learning.py contains the code to run active learning loop and SVM classifier. It is used in the ML tasks in Note 10, Note 11, and Note 12 of Supplementary Information.

2.run_cross_validation.py contains the code to run k-fold cross validation of a model. If data agumentation is used, it is done during the cross validation process. Data agumentation is discussed in paper's supporting information Note S7.

3.run_ga_combination.py contains the code to run GA selection of decision programs. It is discussed in paper's Supplementary Information Note S7.

4.run_hyparam_opt.py contains the code to run hyperparameter optimization of the model.

5.run_inverse_design.py contains the code to run inverse strain sensor design. It is discussed in paper's Supplementary Information Note S9.

6.run_pearson.py contains the code to run Pearson’s coefficient analysis. It is discussed in paper's Supplementary Information Note S8.

7.run_preprocessing.py contains the code to run pre-processing of the raw data.

8.run_svm_classifier.py contains the code to get the SVM classifier. It is discussed in paper's Supplementary Information Note S3.

## Final prediction model ##
Here is the link for the prediction model with the best prediction performance (named as prediction model (UIP+GA) in the paper). It is an ensemble of 16 decision programs with different hyperparameters. And each decision program is an average from 10-fold cross validation. As a result, the total number of h5 files is 160. 
https://drive.google.com/drive/folders/1_SV7zQqVZU8iN6DPhjl1X_01NympHnbY

## Demo ##
* SVM classifier training
The required training data is prepared in ./demo/grid file. The file is in pickle file form, it is produced by preprocessing the raw data grid_data.xlsx with our preprocessing script. 

If you want to train a svm classifier from our data, you can use the following code. Please make sure you have created a results directory. 
```
python run_demo_svm_training
```
It will produce an excel containing the performance of SVMs and their corresponding hyparameters in your result directory. After that, you could choose the best hyperparameter and change the case to case = 2 in the script to train a specific svm model and save it.
