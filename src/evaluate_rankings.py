

import os
import logging
import argparse
import itertools
import pandas as pd
import numpy as np
from nltk.metrics.agreement import AnnotationTask

def nltk_with_kippendorff_data():
    # needs data to be shaped in triples: (coder,item,label)

    input_eval_dp = "../data/krippendorff-evaluation-dataset.csv"

    rcsi_eval_df = pd.read_table(input_eval_dp, delimiter=',', index_col=0)
    print(rcsi_eval_df.head())

    # reshape rcsi data
    rcsi_eval_nltk_df = pd.DataFrame()
    for index, row in rcsi_eval_df.iterrows():
        rcsi_eval_nltk_df = rcsi_eval_nltk_df.append({'coder': 'obs_1', 'item': index, 'label': row['obs1']},
                                                     ignore_index=True)
        rcsi_eval_nltk_df = rcsi_eval_nltk_df.append({'coder': 'obs_2', 'item': index, 'label': row['obs2']},
                                                     ignore_index=True)
        rcsi_eval_nltk_df = rcsi_eval_nltk_df.append({'coder': 'obs_3', 'item': index, 'label': row['obs3']},
                                                     ignore_index=True)
        rcsi_eval_nltk_df = rcsi_eval_nltk_df.append({'coder': 'obs_4', 'item': index, 'label': row['obs4']},
                                                     ignore_index=True)
        rcsi_eval_nltk_df = rcsi_eval_nltk_df.append({'coder': 'obs_5', 'item': index, 'label': row['obs5']},
                                                     ignore_index=True)
    print(rcsi_eval_nltk_df)

    annotation_triples = rcsi_eval_nltk_df.values.tolist()
    # print(annotation_triples)

    t = AnnotationTask(annotation_triples)

    print("alpha:\t\t\t\t\t", t.alpha())
    # print("kappa:\t\t\t\t\t", t.kappa())
    # print("kappa_pairwise:\t\t\t", t.kappa_pairwise('obs1', 'obs2'))
    # print("multi_kappa:\t\t\t", t.multi_kappa())
    # print("weighted_kappa:\t\t\t ", t.weighted_kappa())
    # print("weighted_kappa_pairwise: ", t.weighted_kappa_pairwise('obs1', 'obs2'))
    # print("pi:\t\t\t\t\t\t", t.pi())
    # print("S:\t\t\t\t\t\t ", t.S())


if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    nltk_with_kippendorff_data()

