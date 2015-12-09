import numpy as np
import pandas as pd
from scipy import sparse
import sklearn as skl

__author__ = 'Devin Denis'

f = open("train.json")
whats_cooking = pd.read_json(f)


def get_all_ingredients(data):
    all_ingredients = set()
    for recipe_ingredients in data["ingredients"]:
        for ingredient in recipe_ingredients:
            all_ingredients.add(ingredient)
    return all_ingredients


def create_ingredient_bool_columns(data):
    all_ingredients = get_all_ingredients(data)
    for ingredient in all_ingredients:
        data[ingredient] = data["ingredients"] == ingredient


def main():
    create_ingredient_bool_columns(whats_cooking)
    coo_sparse_ing = sparse.coo_matrix(whats_cooking.iloc[:, 2:])




if __name__ == '__main__':
    main()
