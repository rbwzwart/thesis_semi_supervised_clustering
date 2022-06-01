import pandas as pd
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
from active_semi_clustering.active.pairwise_constraints import ExpertOracle, ExploreConsolidate, MinMax
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score


"Prepare data"
data = pd.read_csv("final_dataset.csv")
labels = pd.read_csv("cluster_labels.csv", delimiter=";")
labels = labels.drop_duplicates(subset="session")
df = data.iloc[: , 1:]
df = df.to_numpy()
scaler = MinMaxScaler()
df = scaler.fit_transform(df)
pca = PCA(n_components=2)
df_scatter = pca.fit_transform(df)

labels = labels[["session", "label_nr"]]
labels = pd.merge(data, labels, on="session")
labels_array = np.array(labels["label_nr"])

encoder = load_model('encoder_C402040.h5')
X = encoder.predict(df)

"Start active learning and fit model"
queries = [50, 100, 150, 200]
weights = [0.5, 1, 1.5, 2]
best_query = 0
best_weight = 0
best_ami = 0
best_results = 0

"Fit PCKMeans models and save best results"
for query in queries:
    np.random.seed(0)
    oracle = ExpertOracle(labels=labels_array, max_queries_cnt=query)
    active_learner = MinMax(n_clusters=5)
    active_learner.fit(X, oracle=oracle)
    pairwise_constraints = active_learner.pairwise_constraints_

    for weight in weights:
        clusterer = PCKMeans(n_clusters=5, max_iter=100, w=weight)
        clusterer.fit(X, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
        results_semi = clusterer.labels_
        data["PCKmeans_exp_result"] = np.array(results_semi)
        label = labels["label_nr"]
        AMI_score = adjusted_mutual_info_score(data["PCKmeans_exp_result"], label)
        print("PCKMEANS")
        print("\n")
        print("Query_size: ", query)
        print("Weight_size", weight)
        print("AMI_score", AMI_score)
        if AMI_score > best_ami:
            print("NEW BEST")
            best_query = query
            best_weight = weight
            best_ami = AMI_score
            best_results = results_semi
            print("Best_AMI: ", AMI_score)
            print("Best_query: ", query)
            print("Best_weight", weight)




"Create plot of best results"
plt.rcParams["figure.figsize"] = (12, 7)
palette = sns.color_palette(cc.glasbey, n_colors=5)
sns.set_style("darkgrid")
sns.scatterplot(df_scatter[:,0],
                df_scatter[:,1],
                hue=best_results,
                data=df,
                palette=palette,
                alpha=0.4,
                s=45)

plt.legend(ncol=1)
plt.title("PCKmeans")
plt.savefig('PCKMeans_scatter.png', bbox_inches='tight')
plt.show()

describe_df = pd.DataFrame(best_results)
print(describe_df.value_counts())

"Save best results"
data["PCKmeans_exp_result"] = np.array(best_results)
data = data[["session", "PCKmeans_exp_result"]]
df2 = pd.read_csv("expert_dataset.csv")
df2 =  pd.merge(data, df2, on="session")
df2.to_csv("results_PCKmeans_exp.csv", index=False)

exit()


