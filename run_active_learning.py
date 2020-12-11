from own_package.active_learning import acquisition_opt, l2_points_opt
from own_package.others import create_results_directory

bounds = [[0, 1],
          [0, 1],
          [200, 2000],
          [0, 2]]


def selector(run, **kwargs):
    if run == 1:
        write_dir = kwargs['write_dir']
        skopt_params = {'total_run': 15, 'random_run': 2}
        psoga_params = {'c1': 1.5, 'c2': 1.5, 'wmin': 0.4, 'wmax': 0.9,
                        'ga_iter_min': 2, 'ga_iter_max': 10, 'iter_gamma': 10,
                        'ga_num_min': 5, 'ga_num_max': 20, 'num_beta': 15,
                        'tourn_size': 3, 'cxpd': 0.9, 'mutpd': 0.05, 'indpd': 0.5, 'eta': 0.5,
                        'pso_iter': 3, 'swarm_size': 10}
        acquisition_opt(bounds=bounds, write_dir=write_dir,
                        svm_directory='./results/svm_classifier/gamma130/models',
                        loader_file='./excel/Data_loader_Round12.xlsx',
                        opt_mode='psoga', opt_params={**skopt_params, **psoga_params},
                        batch_runs=8,
                        normalise_labels=False,
                        norm_mask=[0, 1, 3, 4, 5],
                        ignore_distance=False)  # If true, means use variance only
    elif run == 2:
        numel = kwargs['numel']
        svm_store = kwargs['svm_store']
        seed_number_expt = kwargs['seed_number_expt']
        total_expt = kwargs['total_expt']
        write_dir = kwargs['write_dir']
        l2_points_opt(numel=numel, write_dir=write_dir, svm_directory=svm_store, l2_opt=False,
                      seed_number_of_expt=seed_number_expt, total_expt=total_expt)

selector(1, write_dir='./results/active_learning/Round12')
