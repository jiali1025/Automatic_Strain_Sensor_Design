from scipy.stats import spearmanr
import pandas as pd
'''
This is the code for analysing the spearman coefficient for the data, we only look at the max eplison
'''
df = pd.read_excel('./excel/2D_spearman.xlsx', engine='openpyxl')
df_np = df.to_numpy()

# prepare data
data_CNT = df_np[:,0]
data_PVA = df_np[:,1]
data_thickness = df_np[:,2]
data_end = df_np[:,3]
# calculate spearman's correlation
coef_1, p_1 = spearmanr(data_thickness, data_end)
print('Spearmans correlation coefficient: %.3f' % coef_1)
# interpret the significance
alpha = 0.05
if p_1 > alpha:
	print('Samples are uncorrelated (fail to reject H0) p_1=%.3f' % p_1)
else:
	print('Samples are correlated (reject H0) p_1=%.3f' % p_1)

print('cool')