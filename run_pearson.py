from own_package.analysis import features_pearson_analysis


def selector(case):
    if case == 1:
        # Note: Use seaborn 0.10.1 version
        features_pearson_analysis(data_excel='./excel/Data_loader_Round13.xlsx',
                                  results_directory='./results/pearson/pearson_results')


selector(1)
