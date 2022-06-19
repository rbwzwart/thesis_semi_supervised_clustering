import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score
import seaborn as sns


"""""
A combined set of all command data, the labels and the predicted labels by the best models are combined here.
This allows for easy inspection of the full sessions and how they are clustered
"""""

"Combine labeled set and master set"
df = pd.read_csv("cluster_labels.csv", delimiter=";")
df = df.iloc[: , 1:]
del df["duration"]
df2 = pd.read_csv("final_dataset.csv")
df2 = df2[["session", "duration"]]
df = pd.merge(df, df2, on="session")

"Add all clustering results"
seedK =  pd.read_csv("results_SeededKmeans.csv")
seedK = seedK.iloc[: , :-4]
seedK = seedK.drop_duplicates(subset="session")

kmeans = pd.read_csv("results_KMeans.csv")
kmeans = kmeans.iloc[: , :-4]
kmeans = kmeans.drop_duplicates(subset="session")

PCK_exp = pd.read_csv("results_PCKmeans_exp.csv")
PCK_exp = PCK_exp.iloc[: , :-4]
PCK_exp = PCK_exp.drop_duplicates(subset="session")

df2 = pd.merge(df, seedK, on="session")
df2 = pd.merge(df2, kmeans, on="session")
df2 = pd.merge(df2, PCK_exp, on="session")

"Save for later inspection"
df2.to_csv("combined_results.csv")


"""
The rest of this code was written for inspection and plotting of the results of each of the models.
"""
df2 = df2.drop_duplicates(subset="session")
print(df2.columns)
print(df2["label_desc"].value_counts())
print("\n")


"""
Assigned attack labels:

ssh infiltration         890
busybox attack           472
system reconnaissance    382
malware attack           329
file transfer             42
"""


"Show amount of labeled session per cluster of best models"
result_columns = ["SeededKmeans_result", "Kmeans_results", "PCKmeans_exp_result"]
clusters = [0, 1, 2, 3, 4]

for result in result_columns:
    for cluster in clusters:
        count_df = df2[df2[result] == cluster]
        print(result, cluster)
        print("\n")
        print(count_df["label_desc"].value_counts())
        print("\n")
        print(count_df["duration"].describe())
        print("\n")
        print(count_df["command_freq"].describe())
        print("\n")
        print("\n")


"Calculate AMI of best scores again for each model"
print("AMI PCKMeans:", adjusted_mutual_info_score(df2["label_nr"],df2["PCKmeans_exp_result"]))
print("AMI KMeans:",adjusted_mutual_info_score(df2["label_nr"],df2["Kmeans_results"]))
print("AMI SeededKMeans:",adjusted_mutual_info_score(df2["label_nr"],df2["SeededKmeans_result"]))




"Plot PCKMeans AMI for all models"
sns.set_style("dark")
pck_dict = {
    "50": {'0.5': 0.7798322576472341, '1': 0.8063520134027355, '1.5':0.8377438842625824, '2':0.8815146608590781},
    "100": {'0.5':0.8230968096268618, '1': 0.8279810895228491, '1.5':0.8525415457313668, '2':0.8255263336625277},
    "150": {'0.5':0.8298429103536636, '1': 0.8261144126325872, '1.5':0.8925671808692188, '2':0.8705660386246707},
    "200": {'0.5':0.8351183425882895, '1': 0.831549104698685, '1.5':0.8300905879496976, '2':0.8949434943324779},
}

# df_plot = pd.DataFrame(pck_dict)
# ax = df_plot.T.plot.barh(figsize=(17,10), rot=0, width=0.9)
# ax.tick_params(axis='x', labelsize=22)
# ax.tick_params(axis='y', labelsize=22)
# ax.set_xlabel("AMI Score", fontdict={'fontsize':24})
# ax.set_ylabel("Maximum Queries", fontdict={'fontsize':24})
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(reversed(handles), reversed(labels), fontsize = 15, title="Penalty size", prop={'size': 22}, title_fontsize=22, loc='upper left')
# ax.set_xlim(.4, 1)
# for container in ax.containers:
#     ax.bar_label(container, fontsize=22)
# plt.savefig('PCKMeans_results.png', bbox_inches='tight')
# plt.show()


# "Plot SeededKMeans AMI for all models"
#
# SeedK_dict = {
#     "50": [0.8440299986219179],
#     "100": [0.8621850090288193],
#     "150": [0.8621850090288193],
#     "200": [0.9259195738517745],
# }
#
# df_plot = pd.DataFrame(SeedK_dict)
# ax = df_plot.T.plot.barh(figsize=(17,7), rot=0, legend=False, width=0.3)
# ax.set_xlim(.4, 1)
# ax.tick_params(axis='x', labelsize=23)
# ax.tick_params(axis='y', labelsize=23)
# ax.set_xlabel("AMI Score", fontdict={'fontsize':25})
# ax.set_ylabel("Sample size", fontdict={'fontsize':25})
# ax.bar_label(ax.containers[0], fontsize=23)
# plt.savefig('SeededKMeans_results.png', bbox_inches='tight')
# plt.show()
#
#
#
"Plot KMeans AMI for all models"

KMeans_dict = {
    "5": [0.8503377927133339],
    "10": [0.7668057528052642],
    "15": [0.7668057528052642],
    "20": [0.7668057528052642],
}

df_plot = pd.DataFrame(KMeans_dict)
ax = df_plot.T.plot.barh(figsize=(17,7), rot=0, legend=False, width=0.3)
ax.set_xlim(.4, 1)
ax.tick_params(axis='x', labelsize=23)
ax.tick_params(axis='y', labelsize=23)
ax.set_xlabel("AMI Score", fontdict={'fontsize':25})
ax.set_ylabel("N_iterations", fontdict={'fontsize':25})
ax.bar_label(ax.containers[0], fontsize=23)
plt.savefig('KMeans_results.png', bbox_inches='tight')
plt.show()




# "Plot best AMI for all models"
#
# combined_AMI_dict = {
#     "KMeans": [0.8503377927133339],
#     "PCKMeans": [0.8949434943324779],
#     "SeededKMeans": [0.9259195738517745],
# }
#
# df_plot = pd.DataFrame(combined_AMI_dict)
# ax = df_plot.T.plot.barh(figsize=(20,7), rot=0, legend=False, width=0.4)
# ax.tick_params(axis='x', labelsize=24)
# ax.tick_params(axis='y', labelsize=24, rotation=40)
# ax.set_xlabel("AMI Score", fontdict={'fontsize':25})
# ax.set_ylabel("Model", fontdict={'fontsize':25})
# ax.bar_label(ax.containers[0], fontsize=26)
# ax.set_xlim(.4, 1)
# plt.savefig('combined_results.png', bbox_inches='tight')
# plt.show()
#


exit()

