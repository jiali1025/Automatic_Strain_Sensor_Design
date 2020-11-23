from own_package.features_labels_setup import load_data_to_fl, load_testset_to_fl
from own_package.models import create_hparams
from own_package.cross_validation import run_kf
from own_package.others import create_results_directory


def selector(case, **kwargs):
    if case == 1:
        # Run normal KF cross validation for a single hyperparameter
        hparams = create_hparams(loss='mse', learning_rate=0.001, reg_l1=0.0005, reg_l2=0., nodes=10,
                                 max_depth=5, num_est=5,
                                 activation='relu', epochs=100, batch_size=16, verbose=0)
        model_mode = 'dtr'
        k_folds = 3
        fl_dir = './excel/Data_loader_Round13.xlsx'
        other_names = ['ett30', 'ettI01']
        other_dir = ['./excel/ett30.xlsx', './excel/ett30I01.xlsx']
        # Load main training data
        fl = load_data_to_fl(fl_dir, normalise_labels=False, norm_mask=[0, 1, 3, 4, 5])
        fl_store = fl.create_kf(k_folds=k_folds, shuffle=True)
        # Load other data to evaluate the model on. e.g. the separate test set
        other_fl_dict = {k: load_testset_to_fl(v, norm_mask=[0, 1, 3, 4, 5], scaler=fl.scaler) for k, v in
                         zip(other_names, other_dir)}
        write_dir = create_results_directory('./results/kf/kf_results', folders=['models', 'plots']
                                             , excels=['kf_results'])
        write_excel = f'{write_dir}/kf_results.xlsx'
        run_kf(model_mode=model_mode, fl=fl, fl_store=fl_store, hparams=hparams, scoring='mse',
               other_fl_dict=other_fl_dict, write_excel_dir=write_excel,
               save_model_name=f'{write_dir}/models/{model_mode}', plot_name=f'{write_dir}/plots/lr')


selector(1)
