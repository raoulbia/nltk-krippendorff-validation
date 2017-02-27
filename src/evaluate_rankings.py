""" Cohen's kappa coefficient is a statistic which measures inter-rater agreement for qualitative (categorical) items.

It is generally thought to be a more robust measure than simple percent agreement calculation, since κ takes into
account the possibility of the agreement occurring by chance. Source: <https://en.wikipedia.org/wiki/Cohen's_kappa>

http://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

re inter-rater agreement: kappa vs. % agreement see: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/>

"""


import os
import logging
import argparse
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from nltk.metrics.agreement import AnnotationTask

from angen.evaluate.read_write_files import get_rcsi_eval

def three_annots(input_eval_dp):
    logger = logging.getLogger()

    files = os.listdir(input_eval_dp)
    # print(files)
    type_label_list = ['gene', 'protein', 'chemical', 'drug', 'disease']

    for file, type in itertools.product(files, type_label_list):
        # print(file, type)
        fp = os.path.join(input_eval_dp, file)
        # print(fp)

        rcsi_eval_df = get_rcsi_eval(fp)

        # print(rcsi_eval_df.shape)
        # logger.info('%s:\n%s', rcsi_eval_df.shape) #, rcsi_eval_df.head())

        rcsi_eval_typed_df = rcsi_eval_df[rcsi_eval_df['type'] == type]
        # print(rcsi_eval_typed_df.head())
        # print(rcsi_eval_typed_df.shape)

        # remove any rows where at least one NA is found
        rcsi_eval_typed_df = rcsi_eval_typed_df.dropna(subset=['judge_1', 'judge_2'])

        # transform the three score columns into a 15 x 3 matrix
        rankings = rcsi_eval_typed_df.iloc[:, -3:].astype(int).as_matrix()
        # print(rankings.head())
        # print(rankings)
        # print(rankings.shape)



        judge_A_matrix = rankings[:, 0] # [rows,cols]
        judge_B_matrix = rankings[:, 1]
        judge_C_matrix = rankings[:, 2]


        # Kappa Cohen 3 annotators
        score = 0
        comb_name = itertools.combinations(['judge_A', 'judge_B', 'judge_C'], 2)
        comb = itertools.combinations([judge_A_matrix, judge_B_matrix, judge_C_matrix], 2)
        for counter, i in zip(enumerate(comb_name), enumerate(comb)):
            # print(counter[1], i[1][0])
            print("\n'{}'\t'{}'\tinter annotator agreement between {}".format(file, type, counter[1]))
            score += cohen_kappa(i[1][0], i[1][1])

        # Kippendorff



        # percent_agreement(rcsi, angen)

def two_annots_by_type_and_biomarker(input_eval_dp):
    logger = logging.getLogger()

    files = os.listdir(input_eval_dp)
    type_label_list = ['gene', 'protein', 'chemical', 'drug', 'disease']

    for file, type in itertools.product(files, type_label_list):
        fp = os.path.join(input_eval_dp, file)
        rcsi_eval_df = get_rcsi_eval(fp)
        rcsi_eval_typed_df = rcsi_eval_df[rcsi_eval_df['type'] == type]

        # remove any rows where at least one NA is found
        rcsi_eval_typed_df = rcsi_eval_typed_df.dropna(subset=['judge_1', 'judge_2'])

        # transform the three columns with scores into a 15 x 3 matrix
        rankings = rcsi_eval_typed_df.iloc[:, -3:-1].astype(int).as_matrix()

        judge_A_matrix = rankings[:, 0]
        judge_B_matrix = rankings[:, 1]

        # Kappa Cohen 2 annotators
        # Kappa Cohen 2 annotators
        print("\n'{}'\tinter annotator agreement between judge A and judge B".format(type))
        cohen_kappa(judge_A_matrix, judge_B_matrix)


def two_annots_by_type(input_eval_dp):
    logger = logging.getLogger()

    files = os.listdir(input_eval_dp)
    type_label_list = ['gene', 'protein', 'chemical', 'drug', 'disease']

    # add all scores from the various files to one DF
    rcsi_eval_df = pd.DataFrame()
    for file in files:
        print(files)
        fp = os.path.join(input_eval_dp, file)
        print(fp)

        tmp_df = get_rcsi_eval(fp)
        rcsi_eval_df = pd.concat([rcsi_eval_df, tmp_df])
        # print(rcsi_eval_df.shape)
    # print(rcsi_eval_df.head())
    rcsi_eval_df = rcsi_eval_df.dropna(subset=['judge_1', 'judge_2'])

    # compute score by type
    for type in type_label_list:

        rcsi_eval_typed_df = rcsi_eval_df[rcsi_eval_df['type'] == type]

        # remove any rows where at least one NA is found
        rcsi_eval_typed_df = rcsi_eval_typed_df.dropna(subset=['judge_1', 'judge_2'])

        # transform the three score columns into a 15 x 3 matrix
        # select the cols with rankings and turn them into `nrows x 2cols` form
        # Note: iloc[:, -3:-1] keeps only judge_1 and judge_2 cols
        # Note: nrows varies depending on nbr of NA rows removed
        rankings_by_type = rcsi_eval_typed_df.iloc[:, -3:-1].astype(int).as_matrix()

        judge_A_matrix = rankings_by_type[:, 0]
        judge_B_matrix = rankings_by_type[:, 1]

        # Kappa Cohen 2 annotators
        print("\n'{}'\tinter annotator agreement between judge A and judge B".format(type))
        cohen_kappa(judge_A_matrix, judge_B_matrix)

    # compute overall score
    rankings_overall = rcsi_eval_df.iloc[:, -3:-1].astype(int).as_matrix()

    judge_A_matrix = rankings_overall[:, 0]
    judge_B_matrix = rankings_overall[:, 1]

    print("\noverall inter annotator agreement:")
    cohen_kappa(judge_A_matrix, judge_B_matrix)






def test_nltk_with_rcsi_data(input_eval_dp):
    # needs data to be shaped in triples: (coder,item,label)

    logger = logging.getLogger()

    rcsi_eval_df = pd.read_table(input_eval_dp, delimiter=',', index_col=0)
    # print(rcsi_eval_df.shape)

    rcsi_eval_df = rcsi_eval_df.dropna(subset=['judge_1', 'judge_2'])
    # print(rcsi_eval_df.head())

    # reshape rcsi data
    rcsi_eval_nltk_df = pd.DataFrame()
    for index, row in rcsi_eval_df.iterrows():
        rcsi_eval_nltk_df = rcsi_eval_nltk_df.append({'coder': 'judge_1', 'item': index, 'label': row['judge_1']},
                                                     ignore_index=True)
        rcsi_eval_nltk_df = rcsi_eval_nltk_df.append({'coder': 'judge_2', 'item': index, 'label': row['judge_2']},
                                                     ignore_index=True)
    # print(rcsi_eval_nltk_df.head())

    annotation_triples = rcsi_eval_nltk_df.values.tolist()
    t = AnnotationTask(annotation_triples)

    print("alpha:\t\t\t\t\t", t.alpha())
    print("kappa:\t\t\t\t\t", t.kappa())
    print("kappa_pairwise:\t\t\t", t.kappa_pairwise('judge_1', 'judge_2'))
    print("multi_kappa:\t\t\t", t.multi_kappa())
    print("weighted_kappa:\t\t\t ", t.weighted_kappa())
    print("weighted_kappa_pairwise: ", t.weighted_kappa_pairwise('judge_1', 'judge_2'))
    print("pi:\t\t\t\t\t\t", t.pi())
    print("S:\t\t\t\t\t\t ", t.S())


def test_nltk_with_kippendorff_data(input_eval_dp):
    # needs data to be shaped in triples: (coder,item,label)

    logger = logging.getLogger()

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
    print(annotation_triples)
    # t = AnnotationTask(annotation_triples)
    #
    # print("alpha:\t\t\t\t\t", t.alpha())
    # print("kappa:\t\t\t\t\t", t.kappa())
    # print("kappa_pairwise:\t\t\t", t.kappa_pairwise('judge_1', 'judge_2'))
    # print("multi_kappa:\t\t\t", t.multi_kappa())
    # print("weighted_kappa:\t\t\t ", t.weighted_kappa())
    # print("weighted_kappa_pairwise: ", t.weighted_kappa_pairwise('judge_1', 'judge_2'))
    # print("pi:\t\t\t\t\t\t", t.pi())
    # print("S:\t\t\t\t\t\t ", t.S())



# cohen kappa
def cohen_kappa(true, pred):
    logger = logging.getLogger()
    score = cohen_kappa_score(true, pred)
    # logger.info("cohen kappa score: %s",score)
    print("cohen kappa score: {}".format(score))
    return score

# percent agreement
def percent_agreement(true, pred):
    logger = logging.getLogger()
    agrees = 0
    for rank_true, rank_pred in zip(true, pred):
        if rank_true == rank_pred:
            agrees += 1
    score = agrees / len(pred)
    print("percent agreement: {}".format(score))
    return score


"""
    examples argument values for commandline:

    --input-eval-dp
    ${TOMOE_DATA_HOME}/data-wp-tomoe-angiogenesis/evaluation/randomised_lists/rcsi/merged

    ${TOMOE_DATA_HOME}/data-external/krippendorff/krippendorff-evaluation-dataset.csv

"""
if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--input-eval-dp', required=True)


    args = argparser.parse_args()

    # three_annots(**vars(args))
    # two_annots_by_type_and_biomarker(**vars(args))
    # two_annots_by_type(**vars(args))
    # test_nltk_with_rcsi_data(**vars(args))
    test_nltk_with_kippendorff_data(**vars(args))

# rcsi = [1, 0, 2, -3, 0, 1]
# angen = [0, 0, 2, -3, 0, 2]
# rcsi = [1, 1, 2, -3, 0, 1]
# angen = [1, -1, 2, -3, 0, 1]
# cohen_kappa(rcsi, angen)
# percent_agreement(rcsi, angen)