import numpy as np
import pandas as pd
from scipy import sparse
from scipy import io as spio
import sklearn.linear_model as sklm
import sklearn.cross_validation as skcv

__author__ = 'Devin Denis'

f = open("train.json")
whats_cooking = pd.read_json(f)

# To speed up testing
whats_cooking = whats_cooking.iloc[0:500, :]

def get_all_ingredients():
    all_ingredients = set()
    for recipe_ingredients in whats_cooking["ingredients"]:
        for ingredient in recipe_ingredients:
            all_ingredients.add(ingredient)
    return all_ingredients


def create_ingredient_bool_columns():
    # http://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies
    print whats_cooking["ingredients"].head(10)
    applied_series = whats_cooking["ingredients"].apply(pd.Series)  # Converts lists to dataframe
    print applied_series.head(10)
    stacked_series = applied_series.stack()  # Puts everything in one column again, creating a multi-level index
    print stacked_series.head(10)
    basic_dummies = pd.get_dummies(stacked_series)  # Gets indicator variables from the categories
    print basic_dummies.head(10)
    summed_dummies = basic_dummies.sum(level=0)  # Not sure if I need this.  Re-merges rows.
    print summed_dummies.head(10)
    # all_ingredients = get_all_ingredients()
    # for ingredient in all_ingredients:
    #     ing_column = whats_cooking["ingredients"] == ingredient
    #     whats_cooking[ingredient] = ing_column


def create_csc_sparse_ing():
    create_ingredient_bool_columns()
    relevant_cols = whats_cooking.iloc[:, 3:].astype(int)

    coo_sparse_ing = sparse.coo_matrix(relevant_cols)
    csr_sparse_ing = coo_sparse_ing.tocsr()

    spio.mmwrite("csr_sparse_ing.mtx", csr_sparse_ing)
    return csr_sparse_ing


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
        csr_sparse_ing = create_csc_sparse_ing()

    cuisine_mapping = get_cuisine_int_mapping()  # Keep the mapping so we can reconstruct them later if we want

    map_cuisines_to_nums(cuisine_mapping)

    y_true = whats_cooking["cuisine"].astype(int)
    x_dense = csr_sparse_ing.todense()

    print x_dense.max()
    print x_dense.min()

    # alg = sklm.Lasso()
    alg = sklm.LogisticRegression(penalty='l1', C=0.1, fit_intercept=False, multi_class='ovr')

    alg.fit(csr_sparse_ing, y_true)

    print "what now"
    # scores = skcv.cross_val_score(alg, csr_sparse_ing, y_true, cv=3)

    # print scores.mean()

if __name__ == '__main__':
    main()
