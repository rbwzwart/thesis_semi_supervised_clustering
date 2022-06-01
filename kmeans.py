from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.decomposition import PCA
import seaborn as sns
import colorcet as cc
from sklearn.metrics import adjusted_mutual_info_score

"Prepare data and get ground truth label of each session"
df = pd.read_csv("final_dataset.csv")
df2 = pd.read_csv("cluster_labels.csv", delimiter=";")
df2 = df2.drop_duplicates(subset="session")
df2 = pd.merge(df, df2, on="session")
ground_truth = df2["label_nr"]
X = df.iloc[: , 1:]

X = X.to_numpy()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

encoder = load_model('encoder_C402040.h5')
X1 = encoder.predict(X)

pca = PCA(n_components=10)
X = pca.fit_transform(X)

"Run KMeans model with different values for n_init and save best model"
k=5
inits = [5, 10, 15, 20]
best_AMI = 0
best_init = 0
best_results = 0
for init in inits:
    model = KMeans(n_clusters=k,
                   init='k-means++',
                   max_iter=100,
                   n_init=init,
                   random_state=0)

    model.fit(X1)
    results = model.predict(X1)
    ami_score = adjusted_mutual_info_score(results, ground_truth)
    print("n_init", init)
    print("AMI Score Kmeans: ", adjusted_mutual_info_score(results, ground_truth))
    print("\n")
    if ami_score > best_AMI:
        print("NEW BEST")
        print("Best AMI score Kmeans: ", ami_score)
        print("Best initializations: ", init)
        print("\n")
        best_init = init
        best_AMI = ami_score
        best_results = results



"Plot results of best model"
plt.rcParams["figure.figsize"] = (12, 7)
sns.set_style("darkgrid")
palette = sns.color_palette(cc.glasbey, n_colors=k)
sns.scatterplot(X[:,0],
                X[:,1],
                s=45,
                hue=best_results,
                alpha=0.4,
                palette=palette)
plt.title("Kmeans")
plt.legend()
plt.savefig('Kmeans_scatter.png', bbox_inches='tight')
plt.show()


"Save results"
df["Kmeans_results"] = best_results
data = df[["session", "Kmeans_results"]]
df2 = pd.read_csv("expert_dataset.csv")
df2 =  pd.merge(data, df2, on="session")
df2.to_csv("results_KMeans.csv", index=False)








