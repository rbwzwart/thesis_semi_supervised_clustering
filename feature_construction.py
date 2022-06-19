from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import seaborn as sns

"Prepare data"
plt.rcParams["figure.figsize"] = (20, 10)
df = pd.read_csv("final_dataset.csv")
df = df.iloc[: , 1:]
df = df.to_numpy()

kf = KFold(n_splits=8, shuffle=True, random_state=0)
scaler = MinMaxScaler()


"Define average to average list of rMSE"
def Average(lst):
    return sum(lst) / len(lst)


""""

The code below was used for calculating the rMSE of all autoencoder models. Every autoencoder composition was 
run seperately and results were saved in the mse_dict variable in this code. A seperate autoencoder was run using 
the best composition, which is why the autoencoder used for testing of the compositions is commented out.

"""


"""
Autoencoder instructions:

For a single hidden layer autoencoder, comment out the encoder and decoder hidden layers and let bottleneck receive
from input_data_shape. Then let the output layer receive input from the bottleneck.

"""
# lowest_mse = 10
# reconstruction_error_list = []
# for train_index, test_index in kf.split(df):
#     X_train, X_test = df[train_index], df[test_index]
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.fit_transform(X_test)
#
#     n_inputs = X_train.shape[1] # number of inputs
#
#     # Input layer
#     input_data_shape= Input(shape=(n_inputs,))
#
#     # number of nodes per layer. Example results in a C(40, 20, 40) model
#     n_first = 40 # First hidden encoder layer and second hidden decoder layer
#     n_second = 40 # Second hidden encoder layer and first hidden decoder layer
#     n_bottle = 20 # Bottleneck
#
#     # First encoder layer, comment out for single hidden layer autoencoder
#     encoder= Dense(n_first, activation='tanh')(input_data_shape)
#     encoder = BatchNormalization()(encoder)
#
#     # Second encoder layer, comment out for three hidden layer autoencoder
#     # encoder= Dense(n_second, activation='tanh')(encoder)
#     # encoder= BatchNormalization()(encoder)
#
#     # Bottleneck, change 'encoder' to 'input_data_shape' for single hidden layer autoencoder
#     bottleneck = Dense(n_bottle, activation='linear')(encoder)
#
#     # First decoder layer, comment out for single hidden layer autoencoder
#     decoder = Dense(n_second, activation='tanh')(bottleneck)
#     decoder = BatchNormalization()(decoder)
#
#     # Second decoder layer, comment out for three hidden layer autoencoder
#     # decoder = Dense(n_first, activation='tanh')(decoder)
#     # decoder = BatchNormalization()(decoder)
#
#     # Output layer, change decoder to bottleneck for single layer autoencoder
#     output = Dense(n_inputs, activation='linear')(decoder)
#     # Create model
#     model = Model(inputs=input_data_shape, outputs=output)
#     # Compile model
#     model.compile(optimizer='adam', loss='mse')
#
#
#     # Run model
#     history = model.fit(X_train, X_train, epochs=1000, batch_size=100, validation_data=(X_test, X_test),verbose=False)
#
#     # Calculate reconstruction error and save encoder with lowest mean MSE
#     reconstruction_error = Average(history.history['loss'])
#     reconstruction_error_list.append(reconstruction_error)
#     if Average(reconstruction_error_list) < lowest_mse:
#         encoder = Model(inputs=input_data_shape, outputs=bottleneck)
#         encoder.save('encoder_C402040.h5')
#         lowest_mse = Average(reconstruction_error_list)


"Plot learning curves of each separate autoencoder"
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
#
# plt.xlabel("Epochs", fontsize=20)
# plt.ylabel("rMSE", fontsize=20)
# plt.yticks(fontsize=18)
# plt.xticks(fontsize=18)
# plt.title("C(40, 20, 40)", fontsize=22)
# plt.legend()
# plt.show()
# print(reconstruction_error_list)
#


"Save scores"
mse_dict = {
    "C(20)": [0.0033847968699410556, 0.003755376210436225, 0.0037502193306572736, 0.0035655613378621637, 0.0035300567229278384, 0.00378424976952374, 0.003678146712016314, 0.0035249866642989218],
    "C(15)": [0.005304670453071594, 0.005255301627796144, 0.005214599751401693, 0.005545283754821867, 0.0050575875074137, 0.005385252620093524, 0.005366835678461939, 0.00512648894963786],
    "C(20, 15, 20)": [0.006349528352031484, 0.006345051359385252, 0.006394126758445054, 0.0065017463127151134, 0.00617615582793951, 0.006304130491567776, 0.006714936397504061, 0.006671238732291386],
    "C(20, 10, 20)": [0.007076965784886852, 0.007546073036501184, 0.0074640207132324575, 0.007347233237465844, 0.006741786338854581, 0.007086568341590464, 0.007049479798879475, 0.006964342575753108],
    "C(20, 5, 20)" : [0.009251802049111574, 0.008935789466369897, 0.009346193024422973, 0.00892479915684089, 0.008756304084789009, 0.008888700963463635, 0.009213054529856891, 0.008989067231304944],
    "C(32, 16, 32)" : [0.005254431988345459, 0.005567560902098194, 0.005042284974595532, 0.005348381518269889, 0.005007079829927534, 0.005289816670352593, 0.005336383762187325, 0.0052835060558281835],
    "C(40, 20, 40)" : [0.004419435244752094, 0.004578196317655965, 0.004531141768326052, 0.004407947731553577, 0.0042686471807537605, 0.004443565799039788, 0.004419992961804383, 0.004314891592017375],
    "C(32, 16, 8, 16, 32)": [0.006323058039415627, 0.006555164884775877, 0.006563170825829729, 0.0066107279565185305, 0.006409297301666811, 0.006521463497076184, 0.00636998569406569, 0.006445496764034032],
    "C(32, 20, 10, 20, 32)": [0.005688137770630419, 0.005768944513751194, 0.006261270250426605, 0.005833726223092526, 0.005893139391671866, 0.005888767375610769, 0.005963120794156567, 0.006217193590244278],
    "C(40, 25, 15, 25, 40)": [0.004898356004734524, 0.005088347463519312, 0.005034148301696404, 0.005376843806589022, 0.0049562921677716075, 0.00505760967952665, 0.005014206929015927, 0.004778332916786894],
}

"Print scores"
for c,mse in mse_dict.items():
    print("Average for {} is {}".format(c,round(mean(mse), 5)))

"Plot scores"
plt.rcParams["figure.figsize"] = (23, 10)
x_plot = ["f1", 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']
for key in mse_dict.keys():
    sns.lineplot(x=x_plot, y=mse_dict[key], linestyle='--', label=key, marker='o', linewidth=2.5, markersize=10)

plt.xlabel("Folds", fontsize=27)
plt.ylabel("rMSE", fontsize=27)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=22)
plt.tight_layout()
plt.savefig('autoencoder_rMSE.png', bbox_inches='tight')
plt.show()

"Run final model with best composition"
X = scaler.fit_transform(df)

n_inputs = X.shape[1]  # number of inputs

# Input layer
input_data_shape = Input(shape=(n_inputs,))

# number of nodes per layer. Example results in a C(40, 20, 40) model
n_first = 40  # First hidden encoder layer and second hidden decoder layer
n_second = 40  # Second hidden encoder layer and first hidden decoder layer
n_bottle = 20  # Bottleneck

encoder = Dense(n_first, activation='tanh')(input_data_shape)
encoder = BatchNormalization()(encoder)

bottleneck = Dense(n_bottle, activation='linear')(encoder)

decoder = Dense(n_second, activation='tanh')(bottleneck)
decoder = BatchNormalization()(decoder)

output = Dense(n_inputs, activation='linear')(decoder)

model = Model(inputs=input_data_shape, outputs=output)

model.compile(optimizer='adam', loss='mse')

# Train model and save encoder part
history = model.fit(X, X, epochs=1000, batch_size=100, verbose=False)
encoder = Model(inputs=input_data_shape, outputs=bottleneck)

"""
This is were the best autoencoder composition would be saved. However, for full reproduction of the results
of this thesis it is better to use the already saved autoencoder model, since the that will ensure similar
performance in the clustering models. It is therefore commented out here.
"""

# encoder.save('encoder_C402040.h5')

exit()