import numpy as np
import pandas as pd
from scipy import sparse
from scipy import io as spio
import sklearn.linear_model as sklm
import sklearn.ensemble as sken
import sklearn.cross_validation as skcv
import copy
import os

__author__ = 'Devin Denis'

f = open("train.json")
whats_cooking = pd.read_json(f)

# To speed up testing
# whats_cooking = whats_cooking.iloc[0:5000, :]


def get_all_ingredients():
    all_ingredients = set()
    for recipe_ingredients in whats_cooking["ingredients"]:
        for ingredient in recipe_ingredients:
            all_ingredients.add(ingredient)
    return all_ingredients


def get_ingredient_bool_columns():
    # http://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies
    applied_series = whats_cooking["ingredients"].apply(pd.Series)  # Converts lists to dataframe
    stacked_series = applied_series.stack()  # Puts everything in one column again, creating a multi-level index
    basic_dummies = pd.get_dummies(stacked_series)  # Gets indicator variables from the categories
    return basic_dummies.sum(level=0)  # Unfolds multi level index to single level (how?)


def create_csr_sparse_ing():
    ing_cols = get_ingredient_bool_columns()

    ing_cols[ing_cols > 1] = 1  # Fix recipes with duplicate ingredients having values >1 for those ingredients

    coo_sparse_ing = sparse.coo_matrix(ing_cols)
    csr_sparse_ing = coo_sparse_ing.tocsr()

    spio.mmwrite("csr_sparse_ing.mtx", csr_sparse_ing)
    return csr_sparse_ing


def create_filtered_csr_ing(csr_sparse_ing, appears_in_more_than):
    x_dense = csr_sparse_ing.todense()
    x_filtered = x_dense[:, x_dense.sum(axis=0).A1 > appears_in_more_than]
    coo_filtered = sparse.coo_matrix(x_filtered)
    csr_filtered = coo_filtered.tocsr()

    spio.mmwrite("csr_filtered_ing" + str(appears_in_more_than) + ".mtx", csr_filtered)
    return csr_filtered


def get_cuisine_int_mapping():
    cuisines = whats_cooking["cuisine"].unique()
    return {cuis: num + 1 for num, cuis in enumerate(cuisines)}


def map_cuisines_to_nums(cuisine_mapping):
    for key, val in cuisine_mapping.items():
        whats_cooking.loc[whats_cooking["cuisine"] == key, 'cuisine'] = val


def run_alg(data, y_true, alg):
    alg.fit(data, y_true)
    scores = skcv.cross_val_score(alg, data, y_true, cv=5)
    return scores.mean()


def load_or_create_matrices():
    try:
        csr_sparse_ing = spio.mmread("csr_sparse_ing.mtx")
    except IOError:
        csr_sparse_ing = create_csr_sparse_ing()

    csr_filtered_ing = []
    for i in np.arange(1, 11):
        try:
            csr_filtered_ing.append(spio.mmread("csr_filtered_ing" + str(i) + ".mtx"))
        except IOError:
            csr_filtered_ing.append(create_filtered_csr_ing(csr_sparse_ing, i))
    return csr_sparse_ing, csr_filtered_ing


def get_log_regs():
    log_regs = []
    for c in [0.1, 0.5, 1.0, 2.0]:
        for penalty in ['l1', 'l2']:
            log_regs.append([sklm.LogisticRegression(penalty=penalty, C=c, fit_intercept=False, multi_class='ovr'),
                             c, penalty])
    return log_regs


def get_rand_fors():
    rand_fors = []
    for trees in [25, 100, 200]:
        for depth in [5, 10, 20]:
            rand_fors.append([sken.RandomForestClassifier(n_estimators=trees, max_depth=depth,
                                                          max_features='sqrt', n_jobs=-1), trees, depth])
    return rand_fors


def record_run(matrix, matrix_index, y_true, alg, alg_name, params):
    f_result = open("results.txt", mode='a')
    f_result.write('Matrix:' + str(matrix_index) + os.linesep)
    f_result.write(alg_name + os.linesep)
    for param in params:
        f_result.write(str(param) + ", ")
    f_result.write(os.linesep + 'Score:' + str(run_alg(matrix, y_true, alg)) + os.linesep)
    f_result.close()


def main():
    csr_sparse_ing, csr_filtered_ings = load_or_create_matrices()

    cuisine_mapping = get_cuisine_int_mapping()  # Keep the mapping so we can reconstruct them later if we want

    map_cuisines_to_nums(cuisine_mapping)

    y_true = whats_cooking["cuisine"].astype(int)

    log_regs = get_log_regs()

    rand_fors = get_rand_fors()

    matrices = csr_filtered_ings + [csr_sparse_ing]

    for mat_indx, matrix in enumerate(matrices):
        for alg_data in log_regs:
            record_run(matrix, mat_indx, y_true, alg_data[0], "logistic regression", alg_data[1:])
        for alg_data in rand_fors:
            record_run(matrix, mat_indx, y_true, alg_data[0], "random forest", alg_data[1:])


if __name__ == '__main__':
    main()
