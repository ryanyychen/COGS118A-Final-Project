import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

NUM_FOLDS = 10
NUM_TRIALS = 5
'''
Function to calculate the error of a classifier

@param X: Data
@param Y: Labels
@param classifier: Classifier to calculate error for

@return e: Error of classifier
'''
def calc_error(X, Y, classifier):
    Y_pred = classifier.predict(X)
    e = 1 - accuracy_score(Y, Y_pred)
    return e

'''
Draws the heatmap comparing each D value's error value

@param errors: List of errors
@param D_list: List of D values
@param N_list: List of N values
'''
def draw_heatmap(training_errors, D_list, N_list):
    plt.figure(figsize = (8,7))
    ax = sns.heatmap(training_errors, annot=True, fmt='.3f',
                     xticklabels=D_list, yticklabels=N_list)
    ax.collections[0].colorbar.set_label("error")
    ax.set(xlabel = r'Depth', ylabel=r'Number of Estimators')
    plt.title(r'Training Error w.r.t Depth and Number of Estimators')
    plt.show()

'''
Train decision tree models with k-fold validation

@param X_train: Training data
@param Y_train: Training labels
@param num_folds: Number of folds for k-fold validation

@return opt_hyperparam_classifiers: Dictionary containing the best classifier for each fold based on lowest training error
    Each entry contains classifier, C, training error, validation error, X_test_fold, Y_test_fold
'''
def train_decision_tree(X_train, Y_train):
    # Train decision tree with k-fold validation
    classifier = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='entropy'))
    D_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    N_list = [10, 25, 50, 100, 200, 300, 400, 500]
    param_grid = {'estimator__max_depth': D_list, 'n_estimators': N_list}

    grid_search = GridSearchCV(classifier, param_grid, cv=NUM_FOLDS)
    grid_search = grid_search.fit(X_train, Y_train)
    opt_classifier = grid_search.best_estimator_

    cross_val_errors = 1 - grid_search.cv_results_['mean_test_score']
    cross_val_errors = cross_val_errors.reshape(len(N_list), len(D_list))
    draw_heatmap(cross_val_errors, D_list, N_list)

    opt_e_training = calc_error(X_train, Y_train, opt_classifier)
    opt_D = grid_search.best_params_['estimator__max_depth']
    opt_N = grid_search.best_params_['n_estimators']

    print(f'Optimal depth: {opt_D}')
    print(f'Optimal number of estimators: {opt_N}')
    print(f'Optimal training error: {opt_e_training}')

    return opt_classifier, opt_D, opt_N, opt_e_training

'''
Function to run the experiment

@param X: Data
@param Y: Labels
@param test_size: Size of test set to be split from the data

@return best_classifiers: dictionary containing all best (lowest training error) classifiers and their corresponding data
'''
def experiment(X, Y, test_size):
    best_classifiers = {}
    for i in range(NUM_TRIALS):
        print(f'Trial {i+1}')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        classifier, opt_D, opt_N, opt_e_training = train_decision_tree(X_train, Y_train)
        opt_e_testing = calc_error(X_test, Y_test, classifier)
        best_classifiers[i] = {
                                'classifier': classifier,
                                'opt_D': opt_D,
                                'opt_N': opt_N,
                                'opt_e_training': opt_e_training,
                                'opt_e_testing': opt_e_testing,
                                'X_train': X_train,
                                'X_test': X_test,
                                'Y_train': Y_train,
                                'Y_test': Y_test,
                               }
    return best_classifiers