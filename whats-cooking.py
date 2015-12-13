import numpy as np
import pandas as pd
from scipy import sparse
from scipy import io as spio
import sklearn.linear_model as sklm

__author__ = 'Devin Denis'

f = open("train.json")
whats_cooking = pd.read_json(f)

# To speed up testing
# whats_cooking = whats_cooking.iloc[0:500, :]

def get_all_ingredients():
    all_ingredients = set()
    for recipe_ingredients in whats_cooking["ingredients"]:
        for ingredient in recipe_ingredients:
            all_ingredients.add(ingredient)
    return all_ingredients


def create_ingredient_bool_columns():
    all_ingredients = get_all_ingredients()
    for ingredient in all_ingredients:
        whats_cooking[ingredient] = whats_cooking["ingredients"] == ingredient


def create_csc_sparse_ing():
    create_ingredient_bool_columns()
    relevant_cols = whats_cooking.iloc[:, 3:].astype(int)

    coo_sparse_ing = sparse.coo_matrix(relevant_cols)
    csr_sparse_ing = coo_sparse_ing.tocsr()

    spio.mmwrite("csr_sparse_ing.mtx", csr_sparse_ing)
    return csr_sparse_ing


def main():
    try:
        csr_sparse_ing = spio.mmread("csr_sparse_ing.mtx")
    except IOError:
        csr_sparse_ing = create_csc_sparse_ing()


if __name__ == '__main__':
    main()
