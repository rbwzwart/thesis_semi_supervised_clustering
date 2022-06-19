import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np
data_reinier = pd.read_csv("cluster_labels_reinier.csv", delimiter=";")
data_gijs = pd.read_csv("cluster_labels_gijs.csv", delimiter=";")
data_reinier = data_reinier.drop_duplicates(subset="session")
data_gijs = data_gijs.drop_duplicates(subset="session")
print(data_gijs.columns)
data_r = data_reinier["label_nr"].tolist()
data_g = data_gijs["label_nr"].tolist()

print(cohen_kappa_score(data_r, data_g))

exit()