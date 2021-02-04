import pickle
from own_package.svm_classifier import svm_hparam_opt, run_classification
from own_package.others import create_results_directory


def selector(case):
    if case == 1:  # Run svm_hparam opt to determine the optimal gamma
        grid_fl_dir = './demo/grid/grid_data'
        svm_hparam_opt(grid_fl_dir=grid_fl_dir, total_run=20, write_excel_dir='./results/svm_hparam_opt.xlsx')

    elif case == 2:  # Run svm_classifier for a particular value of gamma and save those models
        grid_fl_dir = './demo/grid/grid_data'
        results_dir = create_results_directory('./results/svm_classifier/gamma130', folders=['models'])
        run_classification(grid_fl_dir=grid_fl_dir, write_dir=results_dir, gamma=130)

if __name__ == '__main__':

    selector(case=1)
