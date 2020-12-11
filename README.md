# Automatic Strain Sensor Design via Active Learning and Data Augmentation for Soft Machines
This is the code for paper Automatic Strain Sensor Design via Active Learning and Data Augmentation for Soft Machines

own_package contains most of the useful functions used in this project.

Inside own_package there are following scripts:
1.active_learning.py
2.cross_validation.py
3.data_agumentation.py
4.feature_labels_setup.py
5.ga_combination.py
6.hparam_opt.py
7.inverse_design.py
8.models.py
9.others.py
10.preprocessing.py
11.pso_ga.py
12.svm_classifier.py

1.active_learning.py contains the main codes used in each active learning round. Including the loading function to load the trained svm model and ensembled model and the aquisition function to calculate acquisition score during each round. It is discussed in paper's supporting information Note S5.

2.cross_validation.py contains the main codes used to realize k-fold cross validation of model training process with different evaluation metrics such as MSE and MRE.

3.data_agumentation.py contains the main codes used for data agumentation purpose including the code for SMOTE_REG and Invariant method which is discussed in paper's supporting information Note S7.

4.feature_labels_setup.py contains the main codes used to construct the feature_label class. This specific data structure contains a lot of useful attributes that can be used in different tasks in our projects. 

5.ga_combination.py contains the main codes used for ga selection of the final best model inside the paper. It is discussed in paper's supporting information Note S7.

6.hparam_opt.py contains the main codes used for hyperparameter optimization of all the models used in the paper.

7.inverse_design.py contains the main code used for runing inverse design experiment. Give a set of trained model and a target labels, this optimizer determines a list of suitable candidate experimental conditions to achieve those target labels. It is discussed in paper's supporting information Note S9.

8.models.py contains the main code to build the useful model class inside this project.

9.others.py contains other utilizing functions that can be used on demand.

10.preprocessing.py contains the main codes used for data preprocessing inside this project such as read data from excel as well as preliminary processing to convert label into the form we want.

11.pso_ga.py contains the main codes used for particle swarm optimization method.

12.svm_classifier.py contains the main codes for svm classifier. It is discussed in paper's supporting information Note S3.

In order to just conduct the experiments inside our study, one can just utilize all the different run.py inside of reading the own_package. 
There are:

1.run_active_learning.py
2.run_cross_validation.py
3.run_ga_combination.py
4.run_hyparam_opt.py
5.run_inverse_design.py
6.run_preprocessing.py
7.run_svm_classifier.py

1.run_active_learning.py contains the code to run active learning experiments given the trained navigation model and svm classifier. It is used by Note 10, Note 11 and Note 12 in paper's supporting material.

2.run_cross_validation.py contains the code to run k-fold cross validation of a model. If data agumentation is used, it is done during the cross validation process. Data agumentation is discussed in paper's supporting information Note S7.

3.run_ga_combination.py contains the code to run ga selection of models. It is discussed in paper's supporting information Note S7.

4.run_hyparam_opt.py contains the code to run hyperparameter optimization.

5.run_inverse_design.py contains the code to run inverse design. It is discussed in paper's supporting information Note S9.

6.run_preprocessing.py contains the code to run preprocessing of the raw data.

7.run_svm_classifier.py contains the code to get the svm classifier. It is discussed in paper's supporting information Note S3.

