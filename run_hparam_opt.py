from own_package.hparam_opt import hparam_opt, read_hparam_opt_data_store
from own_package.others import create_results_directory
from own_package.features_labels_setup import load_data_to_fl, load_testset_to_fl

def selector(case, **kwargs):
    if case == 1:
        fl = load_data_to_fl(kwargs['loader_excel'], normalise_labels=False, norm_mask=[0, 1, 3, 4, 5])
        if kwargs['data_augmentation'] == 'smote':
            fl_store = fl.fold_smote_kf_augment(numel=kwargs['numel'], k_folds=kwargs['k_folds'])
        elif kwargs['data_augmentation'] == 'invariant':
            fl_store = fl.fold_invariant_kf_augment(numel=kwargs['numel'], k_folds=kwargs['k_folds'])
        else:
            fl_store = fl.create_kf(k_folds=kwargs['k_folds'])

        other_fl_dict = {k: load_testset_to_fl(v, norm_mask=[0, 1, 3, 4, 5], scaler=fl.scaler) for k, v in
                         zip(kwargs['other_names'], kwargs['other_dir'])}
        write_dir = create_results_directory(f'./results/hparam_opt/{kwargs["write_dir_name"]}_'
                                             f'{kwargs["model_mode"]}_{kwargs["scoring"]}',
                                             folders=['models', 'plots', 'data_store'], excels=['hparam_results'])
        hparam_opt(model_mode=kwargs['model_mode'], fl=fl, fl_store=fl_store, other_fl_dict=other_fl_dict,
                   scoring=kwargs['scoring'], total_run=kwargs['total_run'], random_run=kwargs['random_run'],
                   plot_dir=f'{write_dir}/plots', write_dir=write_dir)
    elif case == 2:
        read_hparam_opt_data_store(write_dir='./results/hparam_opt/Round12_ann_mse')



other_names = ['ett30', 'ett30I01']
other_dir = ['./excel/ett30.xlsx', './excel/ett30I01.xlsx']
selector(case=1, loader_excel='./excel/Data_loader_Round13.xlsx', other_names=other_names, other_dir=other_dir,
         write_dir_name='Round13_invariant', k_folds=3, model_mode='ann', scoring='mse', total_run=15, random_run=1,
         data_augmentation='invaiant', numel=3)


