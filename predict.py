
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np


def load_model():
    # Portuguese to English translator
    order_model = pickle.load(open('model/model.sav', 'rb'))
    return order_model

def run(model):
    data = np.load("./data/train.npz")
    X_test = data['arr_0']

    result = {"data": model.predict(X_test)}
    return result

# if     __name__ == "__main__":
#     result = run(load_model())
#     print(result)