import numpy as np
import pandas as pd
from scipy import sparse
from scipy import io as spio
import sklearn.linear_model as sklm
import sklearn.ensemble as sken
import sklearn.cross_validation as skcv

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


def create_filtered_csr_ing(csr_sparse_ing):
    x_dense = csr_sparse_ing.todense()
    x_filtered = x_dense[:, x_dense.sum(axis=0).A1 < 10]
    coo_filtered = sparse.coo_matrix(x_filtered)
    csr_filtered = coo_filtered.tocsr()

    spio.mmwrite("csr_filtered_ing.mtx", csr_filtered)
    return csr_filtered


def get_cuisine_int_mapping():
    cuisines = whats_cooking["cuisine"].unique()
    return {cuis: num + 1 for num, cuis in enumerate(cuisines)}


def map_cuisines_to_nums(cuisine_mapping):
    for key, val in cuisine_mapping.items():
        whats_cooking.loc[whats_cooking["cuisine"] == key, 'cuisine'] = val


def main():
    try:
        csr_sparse_ing = spio.mmread("csr_sparse_ing.mtx")
    except IOError:
        csr_sparse_ing = create_csr_sparse_ing()

    try:
        csr_filtered_ing = spio.mmread("csr_filtered_ing.mtx")
    except IOError:
        csr_filtered_ing = create_filtered_csr_ing(csr_sparse_ing)

    cuisine_mapping = get_cuisine_int_mapping()  # Keep the mapping so we can reconstruct them later if we want

    map_cuisines_to_nums(cuisine_mapping)

    y_true = whats_cooking["cuisine"].astype(int)

    # alg = sklm.LogisticRegression(penalty='l1', C=0.1, fit_intercept=False, multi_class='ovr')
    alg = sken.RandomForestClassifier(n_estimators=50, max_depth=20, max_features="sqrt", n_jobs=4)

    alg.fit(csr_sparse_ing, y_true)
    # alg.fit(csr_filtered_ing, y_true)

    scores = skcv.cross_val_score(alg, csr_sparse_ing, y_true, cv=10)
    # scores = skcv.cross_val_score(alg, csr_filtered_ing, y_true, cv=10)

    print scores.mean()

if __name__ == '__main__':
    main()
