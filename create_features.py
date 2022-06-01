import pandas as pd
import numpy as np
import re

"Prepare dataset"
df = pd.read_csv("full_dataset.csv")
sessions = df["session"]
sessions = set(sessions)
print("Total number of sessions: ", len(sessions))

"Determine session duration and all to all corresponding rows for convenience"
df[df["duration"]==""] = np.NaN
df2 = df[~df['duration'].isnull()]
df2 = df2.sort_values('duration').drop_duplicates('session', keep='last')
df2 = df2[["session", "duration"]]
df = df[~df['input'].isnull()]
df.drop('duration', inplace=True, axis=1)
df = pd.merge(df, df2, on='session', how='left')

"Count amount of commands per session"
df['command_freq'] = df.groupby('session')['session'].transform('count')


"Save dataset with full commands for labeling. This dataset is later manually turned into the cluster_labels.csv set"
df_semi = df[["session", "duration", "command_freq", "input"]]
df_semi.to_csv("expert_dataset.csv")

"Some outliers are already removed here"
df = df[df["input"].str.contains("Accept-Encoding: gzip|Cookie: rememberMe=1|Host: m.blog.naver.com|User-Agent: Mozilla/5.0|User-Agent: libwww-perl/6.58")==False]

"Make dataset copy for assigning binaries for every present feature in feature list"
df2 = df.copy(deep=True)
del df2["duration"]
del df2["command_freq"]

"Feature list"
inputs_list = ['enable',
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

"Make sure no duplicates exist in the feature list"
inputs_list = list(set(inputs_list))

"Assign binaries"
import shlex
for index, row in df2.iterrows():
    input = row["input"]
    input_split = str(input)
    input_split = shlex.quote(input_split)
    input_split = shlex.split(input_split)
    for command in input_split:
        command = re.split('[/;|]', command)
        for command in command:
            command = command.split()
            for command in command:
                for inputs in inputs_list:
                    if inputs == command:
                        df2.at[index,inputs] = 1


"Sum binaries for all rows, set all to one and remove duplicate session entries"
df2 = df2.groupby(['session'], as_index=False).sum()
sessions = df2["session"]
del df2["Unnamed: 0"]
del df2["session"]
df2[df2 > 1] = 1
df2["session"] = sessions
df.drop_duplicates(subset = ["session"])

"Create final feature set"
useful_features = ["session", "duration", "command_freq"]
df = df[useful_features]
df = pd.merge(df, df2, on=['session'])

for feature in inputs_list:
    useful_features.append(feature)

df = df[useful_features]
df = df.drop_duplicates(subset = ["session"])


"Export final dataset"
df.to_csv("final_dataset.csv", index=False)

exit()


