from own_package.ga_combination import ga_opt


def selector(case):
    if case == 1:
        ga_opt(load_dir_store=['./results/hparam_opt/round13_dtr_mse',
                               './results/hparam_opt/round13_ann_mse'],
               hparams={'init': [0.7, 0.3], 'n_gen': 10, 'n_pop': 10})


selector(case=1)
