from own_package.preprocessing import read_excel_data_to_cutoff, read_grid_data, l2_tracker
from own_package.others import create_results_directory


def run_preprocess(select):
    # Selector to choose which code from preprocessing to run.
    # Here, we only change the file directory to choose which excel file is inputted into the various functions
    if select == 1:
        write_dir = create_results_directory('./results/preprocessing/preprocess_round13')
        read_excel_data_to_cutoff(read_excel_file='./excel/Raw_Data_Round13.xlsx',
                                  write_dir=write_dir)

    elif select == 2:
        write_dir = create_results_directory('./results/preprocessing/grid')
        read_grid_data(read_excel_file='./excel/grid.xlsx', write_dir=write_dir)

    elif select == 3:
        l2_tracker(write_excel_dir='./results/preprocessing/ep_l2_var.xlsx',
                   final_excel_loader='./excel/ep_l2_loader_var.xlsx',
                   last_idx_store=[11, 16])


if __name__ == '__main__':
    run_preprocess(3)
