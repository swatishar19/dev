import sys
sys.setrecursionlimit(10000)
import seaborn
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

#'#ff5520'
##00a1ec
cmap = colors.ListedColormap(['#95a5a6','#2ecc71','#e74c3c'])
cbar_ticks=[-1,0,1]
bioassay_data_after_filter = pd.read_csv("filtered_trainingset_devtox_shengdeFDAtoxCAESER_for_bioprofile.csv", index_col=0)
# bioassay_data_after_filter = bioassay_data_after_filter.head(3000)
print(bioassay_data_after_filter.shape)
arr = bioassay_data_after_filter.values
# arr[arr == 0] = 3
plot = seaborn.clustermap(arr)
plot = seaborn.clustermap(bioassay_data_after_filter, cmap='coolwarm', cbar_kws={"ticks": cbar_ticks}, yticklabels=False, xticklabels=False)

plot.savefig("filtered_bioprofile_dev_shengdeFDAtoxCAESER_trainingset.png")
plt.show()
