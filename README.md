# Automatic Strain Sensor Design via Active Learning and Data Augmentation for Soft Machines

## Installation ##
The requirement environment is as follow
* python >= 3.6


## Explaination of Code ##
This is the code for paper Automatic Strain Sensor Design via Active Learning and Data Augmentation for Soft Machines

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

1.active_learning.py contains the main codes used in each active learning round. Including the loading function to load the trained svm model and ensembled model and the aquisition function to calculate acquisition score during each round. It is discussed in paper's supporting information Note S5.

2.analysis.py contains the main code to do statistical analyses based on Pearson’s coefficients. It is discussed in paper's supporting information Note S8.

3.cross_validation.py contains the main codes used to realize k-fold cross validation of model training process with different evaluation metrics such as MSE and MRE.

4.data_agumentation.py contains the main codes used for data agumentation purpose including the code for SMOTE_REG and Invariant method which is discussed in paper's supporting information Note S7.

5.feature_labels_setup.py contains the main codes used to construct the feature_label class. This specific data structure contains a lot of useful attributes that can be used in different tasks in our projects. 

6.ga_combination.py contains the main codes used for ga selection of the final best model inside the paper. It is discussed in paper's supporting information Note S7.

7.hparam_opt.py contains the main codes used for hyperparameter optimization of all the models used in the paper.

8.inverse_design.py contains the main code used for runing inverse design experiment. Give a set of trained model and a target labels, this optimizer determines a list of suitable candidate experimental conditions to achieve those target labels. It is discussed in paper's supporting information Note S9.

9.models.py contains the main code to build the useful model class inside this project.

10.others.py contains other utilizing functions that can be used on demand.

11.preprocessing.py contains the main codes used for data preprocessing inside this project such as read data from excel as well as preliminary processing to convert label into the form we want.

12.pso_ga.py contains the main codes used for particle swarm optimization method.

13.svm_classifier.py contains the main codes for svm classifier. It is discussed in paper's supporting information Note S3.

In order to just conduct the experiments inside our study, one can just utilize all the different run.py inside of reading the own_package. 
There are:

* 1.run_active_learning.py
* 2.run_cross_validation.py
* 3.run_ga_combination.py
* 4.run_hyparam_opt.py
* 5.run_inverse_design.py
* 6.run_pearson.py
* 7.run_preprocessing.py
* 8.run_svm_classifier.py

1.run_active_learning.py contains the code to run active learning experiments given the trained navigation model and svm classifier. It is used by Note 10, Note 11 and Note 12 in paper's supporting material.

2.run_cross_validation.py contains the code to run k-fold cross validation of a model. If data agumentation is used, it is done during the cross validation process. Data agumentation is discussed in paper's supporting information Note S7.

3.run_ga_combination.py contains the code to run ga selection of models. It is discussed in paper's supporting information Note S7.

4.run_hyparam_opt.py contains the code to run hyperparameter optimization.

5.run_inverse_design.py contains the code to run inverse design. It is discussed in paper's supporting information Note S9.

6.run_pearson.py contains the code to run pearson's coefficient analysis. It is discussed in paper's supporting information Note S8.

7.run_preprocessing.py contains the code to run preprocessing of the raw data.

8.run_svm_classifier.py contains the code to get the svm classifier. It is discussed in paper's supporting information Note S3.

