import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (17, 10)
df = pd.read_csv("final_dataset.csv")
palette = sns.color_palette('pastel')

"DATA INSPECTION"
X_commands = df.iloc[: , 3:]
commands_sum = X_commands.sum()

sns.set(font_scale = 1.8)
commands_sum_largest = commands_sum.nlargest(n=10)
sns.barplot(commands_sum_largest.index, commands_sum_largest.values, palette=palette)
plt.show()

commands_sum_smallest = commands_sum.nsmallest(n=10)
sns.barplot(commands_sum_smallest.index, commands_sum_smallest.values, palette=palette)
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
plt.pie(attack_cat, labels=labels, explode=[0.01]*2, autopct="%.1f%%", colors=palette, textprops={'fontsize': 18})
plt.savefig('Pie_plot_categories.png', bbox_inches='tight')
plt.show()

exit()