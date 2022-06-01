import pandas as pd
import numpy as np

class MaximumQueriesExceeded(Exception):
    pass

class OracleMistake(Exception):
    pass


class ExpertOracle:
    def __init__(self, labels, max_queries_cnt=20):
        self.queries_cnt = 0
        self.max_queries_cnt = max_queries_cnt
        self.labels = labels

    def query(self, i, j):
        "Query the researcher (expert) to find out whether i and j should be must-linked"
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1
            return self.labels[i] == self.labels[j]
        else:
            raise MaximumQueriesExceeded


from scipy.spatial.distance import hamming

# class SimilarityOracle:
#     def __init__(self, threshold=.5, max_queries_cnt=20):
#
#         self.queries_cnt = 0
#         self.max_queries_cnt = max_queries_cnt
#         self.threshold = threshold
#         if self.threshold >= 1 or self.threshold <= 0:
#             print("Error: threshold should be a float between 0 and 1!")
#             raise OracleMistake
#
#         df = pd.read_csv("final_dataset.csv")
#         feature_list = ['enable',
#                         'adminpass',
#                         'uniq',
#                         'scp',
#                         'lscpu',
#                         "while",
#                         "do",
#                         'more',
#                         'cut',
#                         'system',
#                         'sh',
#                         'cat',
#                         'cd',
#                         'tftp',
#                         'wget',
#                         'dd',
#                         'rm',
#                         'exit',
#                         'cp',
#                         'uname',
#                         'curl',
#                         'chmod',
#                         'tar',
#                         'while',
#                         'ps',
#                         'grep',
#                         'ls',
#                         'echo',
#                         'dd',
#                         'pkill',
#                         'nproc',
#                         'help',
#                         'busybox',
#                         'ftpget',
#                         'ifconfig',
#                         'start',
#                         'config',
#                         'su',
#                         'free',
#                         'systemctl',
#                         'stop',
#                         'dmidecode',
#                         'head',
#                         'dmesg',
#                         'lspci',
#                         'scp',
#                         'sudo',
#                         'history',
#                         'install',
#                         'crontab',
#                         'printf',
#                         'mkdir',
#                         'wc',
#                         'chpasswd',
#                         'uptime',
#                         'top',
#                         'w',
#                         'config',
#                         "awk",
#                         "id",
#                         "linuxshell",
#                         "bash",
#                         "shell",
#                         "yum",
#                         "var",
#                         "tmp",
#                         "etc",
#                         "mnt",
#                         "root",
#                         "dev",
#                         "bin",
#                         "ssh",
#                         'proc',
#                         "print",
#                         ]
#         sessions = df["session"]
#         features = df[feature_list]
#         self.features = features.to_numpy()
#
#     def query(self, i, j):
#         "Use similarity to find out whether i and j should be must-linked"
#         if self.queries_cnt < self.max_queries_cnt:
#             self.queries_cnt += 1
#             distance = np.count_nonzero(self.features[i] * self.features[j] > 0)
#             commands = np.count_nonzero(self.features[j] > 0)
#             return distance/commands >= self.threshold
#         else:
#             raise MaximumQueriesExceeded
