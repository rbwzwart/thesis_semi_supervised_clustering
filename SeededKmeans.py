import pandas as pd
from active_semi_clustering.semi_supervised.labeled_data import SeededKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_mutual_info_score
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import colorcet as cc
from sklearn.decomposition import PCA

"Create seed set"
data = pd.read_csv("final_dataset.csv")
df2 = pd.read_csv("cluster_labels.csv", delimiter=";")
df2 = df2.drop_duplicates(subset="session")
df = pd.merge(data, df2, on="session")
ground_truth = df["label_nr"]

"Run SeededKmeans using seed set for different sample_sizes and save best model"
sample_sizes = [10, 20, 30, 40]
best_size = 0
best_AMI = 0
best_results = 0

for sample_size in sample_sizes:
    seed = 0
    cluster_1 = df2[df2["label_nr"] == 0].sample(sample_size, random_state=seed)
    cluster_2 = df2[df2["label_nr"] == 1].sample(sample_size, random_state=seed)
    cluster_3 = df2[df2["label_nr"] == 2].sample(sample_size, random_state=seed)
    cluster_4 = df2[df2["label_nr"] == 3].sample(sample_size, random_state=seed)
    cluster_5 = df2[df2["label_nr"] == 4].sample(sample_size, random_state=seed)
    cluster_centers = pd.concat([cluster_1, cluster_2, cluster_3, cluster_4, cluster_5])

    "Merge seed set with full set and save"
    cluster_centers = cluster_centers[["session", "label_nr"]]
    df = pd.merge(data, cluster_centers, on="session", how="outer")
    df['label_nr'] = df['label_nr'].fillna(-1)
    labels = df['label_nr']
    labels = np.array(df['label_nr'])
    labels = labels.astype(int)
    df = df.iloc[:, 1:-1]
    df1 = df[df.isna().any(axis=1)]

    "Prepare data"
    df = df.to_numpy()
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)

    pca = PCA(n_components=2)
    df_scatter = pca.fit_transform(df)

    encoder = load_model('encoder_C402040.h5')
    X = encoder.predict(df)

    model = SeededKMeans(n_clusters=5, max_iter=100)
    seededkmeans = model.fit(X, labels)
    results = seededkmeans.labels_
    AMI_score = adjusted_mutual_info_score(results, ground_truth)
    print("SEEDEDKMEANS")
    print("\n")
    print("Sample_size:", sample_size)
    print("Ami_score:", AMI_score)
    print("\n")
    if AMI_score > best_AMI:
        best_size = sample_size
        best_AMI = AMI_score
        best_results = results
        print("NEW BEST")
        print("Best AMI:", AMI_score)
        print("Best sample size:", sample_size)
        print("\n")



"Plot best results"
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
plt.title("SeededKmeans")
plt.savefig('SeededKMeans_scatter.png', bbox_inches='tight')
plt.show()

"Save best results"
data["SeededKmeans_result"] = np.array(best_results)
data = data[["session", "SeededKmeans_result"]]
df2 = pd.read_csv("expert_dataset.csv")
df2 =  pd.merge(data, df2, on="session")
df2.to_csv("results_SeededKmeans.csv", index=False)

exit()