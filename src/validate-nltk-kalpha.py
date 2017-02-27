

import logging
import argparse
import itertools
import pandas as pd
import numpy as np
from nltk.metrics.agreement import AnnotationTask
import krippendorff_alpha as kalpha


def nltk_with_kippendorff_data():
    # needs data to be shaped in triples: (coder,item,label)

    input_eval_dp = "../data/krippendorff-evaluation-dataset.csv"

    eval_df = pd.read_table(input_eval_dp, delimiter=',', index_col=0)
    print("\ninput data:\n", eval_df.head())

    # reshape rcsi data
    eval_nltk_df = pd.DataFrame()
    for index, row in eval_df.iterrows():
        eval_nltk_df = eval_nltk_df.append({'coder': 'obs_1', 'item': index, 'label': row['obs1']},
                                                     ignore_index=True)
        eval_nltk_df = eval_nltk_df.append({'coder': 'obs_2', 'item': index, 'label': row['obs2']},
                                                     ignore_index=True)
        eval_nltk_df = eval_nltk_df.append({'coder': 'obs_3', 'item': index, 'label': row['obs3']},
                                                     ignore_index=True)
        eval_nltk_df = eval_nltk_df.append({'coder': 'obs_4', 'item': index, 'label': row['obs4']},
                                                     ignore_index=True)
        eval_nltk_df = eval_nltk_df.append({'coder': 'obs_5', 'item': index, 'label': row['obs5']},
                                                     ignore_index=True)
    print("\nreshaped data:\n\n", eval_nltk_df.head())
    print(eval_nltk_df.tail())

    annotation_triples = eval_nltk_df.values.tolist()
    # print(annotation_triples)

    t = AnnotationTask(annotation_triples)
    print("\nKrippendorff alpha as per NLTK:\t", t.alpha(),
          "\n===========================================\n")


def tgrill_with_kippendorff_data():

    input_eval_dp = "../data/krippendorff-evaluation-dataset.csv"

    eval_df = pd.read_table(input_eval_dp, delimiter=',', index_col=0)
    print("\ninput data:\n", eval_df.head())

    data = eval_df.values.T.tolist()

    missing = '.'  # indicator for missing values
    print("\nKalpha nominal metric: %.3f" % kalpha.krippendorff_alpha(data, kalpha.nominal_metric, missing_items=missing))
    print("Kalpha interval metric: %.3f" % kalpha.krippendorff_alpha(data, kalpha.interval_metric, missing_items=missing))


if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    nltk_with_kippendorff_data()
    tgrill_with_kippendorff_data()

