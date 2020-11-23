import numpy as np
from own_package.inverse_design import inverse_design


def selector(case, **kwargs):
    if case == 1:
        bounds = [[0, 1],
                  [0, 1],
                  [200, 2000],
                  [0, 2]]

        def loss_func(targets, pred):
            # Define your own loss function to target the material properties you want
            # GF10, GF100, end
            indicator0 = 1
            if pred[-1] > 30:  # end must be greater than 30
                indicator2 = 0
            else:
                indicator2 = 1e6  # If less than 30, penalize with some large number
            if pred[1] <10:
                indicator1 = 0
            else:
                indicator1 = 1e6
            # [target GF10 - predicted GF10, Y, Z] * [I1, I2, I3] = XI1 + XI2 + XI3
            return np.mean(((targets - pred) * np.array([indicator0, indicator1, indicator2])) ** 2)

        skopt_params = {'total_run': 15, 'random_run': 2}
        pso_params = {'c1': 1.5, 'c2': 1.5, 'wmin': 0.4, 'wmax': 0.9,
                      'ga_iter_min': 2, 'ga_iter_max': 10, 'iter_gamma': 10,
                      'ga_num_min': 5, 'ga_num_max': 20, 'num_beta': 15,
                      'tourn_size': 3, 'cxpd': 0.9, 'mutpd': 0.05, 'indpd': 0.5, 'eta': 0.5,
                      'pso_iter': 3, 'swarm_size': 10}

        inverse_design(targets=np.array([2, 8, 60]), loss_func=loss_func,
                       bounds=bounds, init_guess=None,
                       opt_mode='psoga', opt_params={**pso_params, **skopt_params},
                       model_directory_store=['./results/inverse_design/models/ann_mse',
                                              './results/inverse_design/models/dtr_mse', ],
                       svm_directory='./results/svm_classifier/gamma130/models',
                       loader_file='./excel/Data_loader_Round13.xlsx',
                       write_dir='./results/inverse_design',
                       )


selector(1)
