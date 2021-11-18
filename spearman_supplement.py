from scipy.stats import spearmanr
import pandas as pd
'''
This is the code for analysing the spearman coefficient for the data, we only look at the max eplison
'''
df = pd.read_excel('./excel/cutoff 125 points.xlsx', engine='openpyxl')
df_np = df.to_numpy()

# prepare data
data_10 = df_np[:,1]
data_100 = df_np[:,2]
data_end = df_np[:,3]

# calculate spearman's correlation
coef_1, p_1 = spearmanr(data_100, data_10)
print('Spearmans correlation coefficient: %.3f' % coef_1)
# interpret the significance
alpha = 0.05
if p_1 > alpha:
	print('Samples are uncorrelated (fail to reject H0) p_1=%.3f' % p_1)
else:
	print('Samples are correlated (reject H0) p_1=%.3f' % p_1)

print('cool')