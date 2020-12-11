import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as stat
from own_package.others import create_results_directory

def features_pearson_analysis(data_excel, results_directory):
    write_dir = create_results_directory(results_directory)
    try:
        del mpl.font_manager.weight_dict['roman']
        mpl.font_manager._rebuild()
    except KeyError:
        pass
    sns.set(style='ticks')
    mpl.rc('font', family='Times New Roman')

    df = pd.read_excel(data_excel, index_col=0, sheet_name='features')
    df_labels = pd.read_excel(data_excel, index_col=0, sheet_name='cutoff')
    working_range = df_labels.iloc[:, -1].values - df_labels.iloc[:, -2].values
    df.insert(loc=df.shape[-1] - 3, column='Working Range', value=working_range)
    df1 = df[df.iloc[:, -3] == 1].iloc[:, :-3]
    df2 = df[df.iloc[:, -2] == 1].iloc[:, :-3]
    df3 = df[df.iloc[:, -1] == 1].iloc[:, :-3]

    x_store = ['CNT Mass Percentage', 'PVA Mass Percentage', 'Thickness nm', 'Mxene Mass Percentage']
    mypal = sns.hls_palette(4, l=.3, s=.8)

    for dimension, df in enumerate([df1, df2, df3]):
        df['Mxene Mass Percentage'] = 1 - df.iloc[:, 0] - df.iloc[:, 1]
        for x, color in zip(x_store, mypal):
            plt.close()
            sns.jointplot(x=x, y='Working Range', data=df, alpha=0.3, color=color, stat_func=stat.pearsonr)
            plt.savefig('{}/{}_dim_{}.png'.format(write_dir, x, dimension), bbox_inches='tight')
