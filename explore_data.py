import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (25, 21)
df = pd.read_csv("final_dataset.csv")
palette = sns.color_palette('pastel')

"DATA INSPECTION"
X_commands = df.iloc[: , 3:]
commands_sum = X_commands.sum()
commands_sum_largest = commands_sum.nlargest(n=10)
commands_sum_smallest = commands_sum.nsmallest(n=10)

# sns.set(font_scale = 2.3)
fig, axs = plt.subplots(ncols=2)
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=65, fontsize=30)
    plt.yticks(fontsize=30)

sns.set_style("darkgrid")
ax_1 = sns.barplot(commands_sum_largest.index, commands_sum_largest.values, palette=palette, ax=axs[0])
ax_2 = sns.barplot(commands_sum_smallest.index, commands_sum_smallest.values, palette=palette, ax=axs[1])
start, end = ax_2.get_ylim()
ax_2.yaxis.set_ticks(np.arange(start, end, 1))
start, end = ax_1.get_ylim()
ax_1.yaxis.set_ticks(np.arange(start, end, 400))
ax_1.set_title("Most used commands", fontsize=30)
ax_2.set_title("Least used commands", fontsize=30)

plt.show()

print(df["duration"].describe())
print(df["command_freq"].describe())

"Drop outliers"
df = df.loc[~((df['duration'] > 300)),:]
df = df.loc[~((df['command_freq'] > 30)),:]
feature_list = ['enable',
               'adminpass',
               'uniq',
               'scp',
               'lscpu',
               "while",
               "do",
               'more',
               'cut',
               'system',
               'sh',
               'cat',
               'cd',
               'tftp',
               'wget',
               'dd',
               'rm',
               'exit',
               'cp',
               'uname',
               'curl',
               'chmod',
               'tar',
               'while',
               'ps',
               'grep',
               'ls',
               'echo',
               'dd',
               'pkill',
               'nproc',
               'help',
               'busybox',
               'ftpget',
               'ifconfig',
               'start',
               'config',
               'su',
               'free',
               'systemctl',
               'stop',
               'dmidecode',
               'head',
               'dmesg',
               'lspci',
               'scp',
               'sudo',
               'history',
               'install',
               'crontab',
               'printf',
               'mkdir',
               'wc',
               'chpasswd',
               'uptime',
               'top',
               'w',
               'config',
               "awk",
               "id",
               "linuxshell",
               "bash",
               "print",
               "shell",
               "yum",
               "var",
               "tmp",
               "etc",
               "mnt",
               "root",
               "dev",
               "bin",
               "ssh",
               'proc',
               ".ssh"
               ]
df["sum_features"] = df[feature_list].sum(axis=1)
df = df.loc[(df['sum_features'] > 0),:]
del df["sum_features"]
print(df["duration"].describe())
print(df["command_freq"].describe())
df.to_csv("final_dataset.csv", index=False)

"Find unique sessions"
uniques = df[feature_list]
print(len(uniques[feature_list].drop_duplicates()))
print(uniques.columns)


"Inspect labeled set"
df2 = pd.read_csv("cluster_labels.csv", delimiter=";")
df2 = df2.drop_duplicates(subset="session")
attack_cat = df2["attack_cat"].value_counts()
print(attack_cat)
labels = ["Fingerprinting", "Malicious activity"]
palette = sns.color_palette('pastel')
plt.pie(attack_cat, labels=labels, explode=[0.01]*2, autopct="%.1f%%", colors=palette, textprops={'fontsize': 45})

plt.savefig('Pie_plot_categories.png', bbox_inches='tight')
plt.show()

exit()