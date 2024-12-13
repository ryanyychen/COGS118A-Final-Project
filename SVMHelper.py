import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
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
Draws the heatmap comparing each C value's error value

@param errors: List of errors
@param C_list: List of C values
@param title: Title of the heatmap
'''
def draw_heatmap(errors, D_list, title):
    plt.figure(figsize = (2,4))
    ax = sns.heatmap(errors, annot=True, fmt='.3f', yticklabels=D_list, xticklabels=[])
    ax.collections[0].colorbar.set_label('error')
    ax.set(ylabel='C')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(title)
    plt.show()

'''
Train linear SVM models with k-fold validation

@param X_train: Training data
@param Y_train: Training labels
@param num_folds: Number of folds for k-fold validation

@return opt_hyperparam_classifiers: Dictionary containing the best classifier for each fold based on lowest training error
    Each entry contains classifier, C, training error, validation error, X_test_fold, Y_test_fold
'''
def train_linear_svm(X_train, Y_train):
    C_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

    # Train linear SVM with k-fold validation
    classifier = LinearSVC(loss='hinge', max_iter=1000000)
    param_grid = {'C': C_list}

    grid_search = GridSearchCV(classifier, param_grid, cv=NUM_FOLDS)
    grid_search = grid_search.fit(X_train, Y_train)
    opt_classifier = grid_search.best_estimator_
    
    cross_val_errors = 1 - grid_search.cv_results_['mean_test_score'].reshape(-1,1)
    draw_heatmap(cross_val_errors, C_list, title='cross-validation error w.r.t C')

    opt_e_training = calc_error(X_train, Y_train, opt_classifier)
    opt_C = grid_search.best_params_['C']

    print(f'Optimal C: {opt_C}')
    print(f'Optimal training error: {opt_e_training}')
    print(f'Decision Boundary: {opt_classifier.coef_[0][0]}x0 + {opt_classifier.coef_[0][1]}x1 + {opt_classifier.intercept_[0]} = 0')

        
    return opt_classifier, opt_C, opt_e_training

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
        classifier, opt_C, opt_e_training = train_linear_svm(X_train, Y_train)
        opt_e_testing = calc_error(X_test, Y_test, classifier)
        best_classifiers[i] = {
                                'classifier': classifier,
                                'opt_C': opt_C,
                                'opt_e_training': opt_e_training,
                                'opt_e_testing': opt_e_testing,
                                'X_train': X_train,
                                'X_test': X_test,
                                'Y_train': Y_train,
                                'Y_test': Y_test,
                               }
    return best_classifiers

'''
Visualize the validation data and decision boundary for each fold's best classifier

@param classifiers: Dictionary containing the best classifiers for each fold
                    Each entry contains: classifier, C, e_training, e_validation, X_test_fold, Y_test_fold

@output: Subplot containing k plots, where k in the number of folds
'''
def vis_trial(classifier_info, x_label, y_label, pos_label, neg_label):
    classifier = classifier_info['classifier']
    X_train = classifier_info['X_train']
    X_test = classifier_info['X_test']
    Y_train = classifier_info['Y_train']
    Y_test = classifier_info['Y_test']

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot data points in training set
    indices_pos1 = (Y_test == 1).nonzero()[0]
    indices_neg1 = (Y_test == -1).nonzero()[0]
    axs[0].scatter(X_test[:,0][indices_pos1], X_test[:,1][indices_pos1],
                    c='green', label=pos_label)
    axs[0].scatter(X_test[:,0][indices_neg1], X_test[:,1][indices_neg1],
                    c='red', label=neg_label)
    ax[0].legend()
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel(y_label)
    ax[0].set_title(f'Fold {key + 1} Cross-Validation')

    # Plot data points in test set
    indices_pos1 = (Y_test == 1).nonzero()[0]
    indices_neg1 = (Y_test == -1).nonzero()[0]
    plt.scatter(X_test[:,0][indices_pos1], X_test[:,1][indices_pos1],
                    c='green', label=pos_label)
    plt.scatter(X_test[:,0][indices_neg1], X_test[:,1][indices_neg1],
                    c='red', label=neg_label)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    for ax, (key, value) in zip(axs, classifiers.items()):
        X_test = value['X_test_fold']
        Y_test = value['Y_test_fold']
        
        # Plot data points in test set
        indices_pos1 = (Y_test == 1).nonzero()[0]
        indices_neg1 = (Y_test == -1).nonzero()[0]
        ax.scatter(X_test[:,0][indices_pos1], X_test[:,1][indices_pos1],
                    c='green', label=pos_label)
        ax.scatter(X_test[:,0][indices_neg1], X_test[:,1][indices_neg1],
                    c='red', label=neg_label)
        ax.legend()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'Fold {key + 1} Cross-Validation')

        # Plot decision boundary
        W = value['classifier'].coef_[0]
        w0 = W[0]
        w1 = W[1]
        b = value['classifier'].intercept_[0]
        temp = -w1*np.array([X_test[:,1].min(), X_test[:,1].max()])/w0-b/w0
        x0_min = max(temp.min(), X_test[:,0].min())
        x0_max = min(temp.max(), X_test[:,1].max())
        x0 = np.linspace(x0_min,x0_max,100)
        x1 = -w0*x0/w1-b/w1
        ax.plot(x0,x1,color='black')
    plt.show()

# '''
# Summary on conducted experiment
# For each trial:
#     1. Visualize each selected fold-best classifier's decision boundary against validation set
#     2. Pick out the best classifier (lowest validation error)

# @param trial_classifiers: Dictionary containing the best classifiers for each trial
#                         Format: {
#                                 trial_number:
#                                     {
#                                     X_train,
#                                     X_test,
#                                     Y_train,
#                                     Y_test,
#                                     classifier,
#                                     },
#                                 }
# @return opt_validation_classifiers: Dictionary containing the best classifiers for each trial
# '''
# def experiment_summary(trial_classifiers, x_label, y_label, pos_label, neg_label):
#     for i in range(NUM_TRIALS):
#         print(f'Trial {i+1}')
#         print('----------------')

#         # Retrieve information from dictionary
#         classifier = trial_classifiers[i]['classifier']
#         X_train = trial_classifiers[i]['X_train']
#         X_test = trial_classifiers[i]['X_test']
#         Y_train = trial_classifiers[i]['Y_train']
#         Y_test = trial_classifiers[i]['Y_test']

#         vis_trial(trial_classifiers[i], x_label, y_label, pos_label, neg_label)

#         opt_e_validation = 1
#         for (key, value) in classifiers.items():
#             # Print information of optimal classifier for fold
#             print(f'Fold {key+1}')
#             print(f'C: {value['C']}')
#             W = value['classifier'].coef_[0]
#             b = value['classifier'].intercept_[0]
#             print(f'Decision Boundary: {W[0]}x0 + {W[1]}x1+ {b} = 0')
#             print(f'Training Error: {value['e_training']}')
#             print(f'Validation Error: {value['e_validation']}')

#             # Search for optimal classifier in trial (lowest validation error)
#             if (value['e_validation'] < opt_e_validation):
#                 opt_e_validation = value['e_validation']
#                 opt_classifier = value['classifier']
#                 e_training = value['e_training']
#                 C = value['C']
            
#         e_test = calc_error(X_test, Y_test, opt_classifier)

#         opt_validation_classifiers[i] = {
#                                 'classifier': opt_classifier,
#                                 'C': C,
#                                 'e_training': e_training,
#                                 'e_validation': opt_e_validation,
#                                 'e_test': e_test,
#                                 'X_train': X_train,
#                                 'X_test': X_test,
#                                 'Y_train': Y_train,
#                                 'Y_test': Y_test,
#                                 }
#         # Print information of optimal classifier for trial
#         print('----------------')
#         print('Optimal Classifier')
#         print('----------------')
#         print(f'C: {C}')
#         W = opt_classifier.coef_[0]
#         b = opt_classifier.intercept_[0]
#         print(f'Decision Boundary: {W[0]}x0 + {W[1]}x1+ {b} = 0')
#         print(f'Training Error: {e_training}')
#         print(f'Validation Error: {opt_e_validation}')
#         print(f'Test Error: {e_test}')
#     return opt_validation_classifiers

'''
Function to visualize the best classifiers in each experiment, plotted on training data and test data
'''
def vis_experiment(classifier_data, x_label, y_label, pos_label, neg_label):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
    classifier = classifier_data['classifier']
    X_train = classifier_data['X_train']
    Y_train = classifier_data['Y_train']
    X_test = classifier_data['X_test']
    Y_test = classifier_data['Y_test']

    # Plot data points in training set
    indices_pos1 = (Y_train == 1).nonzero()[0]
    indices_neg1 = (Y_train == -1).nonzero()[0]
    axs[0].scatter(X_train[:,0][indices_pos1], X_train[:,1][indices_pos1],
                c='green', label=pos_label)
    axs[0].scatter(X_train[:,0][indices_neg1], X_train[:,1][indices_neg1],
                c='red', label=neg_label)
    axs[0].legend()
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)
    axs[0].set_title(f'Train Data')

    # Plot data points in test set
    indices_pos1 = (Y_test == 1).nonzero()[0]
    indices_neg1 = (Y_test == -1).nonzero()[0]
    axs[1].scatter(X_test[:,0][indices_pos1], X_test[:,1][indices_pos1],
                c='green', label=pos_label)
    axs[1].scatter(X_test[:,0][indices_neg1], X_test[:,1][indices_neg1],
                c='red', label=neg_label)
    axs[1].legend()
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_title(f'Test Data')

    # Plot decision boundary
    W = classifier.coef_[0]
    w0 = W[0]
    w1 = W[1]
    b = classifier.intercept_[0]
    temp = -w1*np.array([X_test[:,1].min(), X_test[:,1].max()])/w0-b/w0
    x0_min = max(temp.min(), X_test[:,0].min())
    x0_max = min(temp.max(), X_test[:,1].max())
    x0 = np.linspace(x0_min,x0_max,100)
    x1 = -w0*x0/w1-b/w1
    axs[0].plot(x0,x1,color='black')
    axs[1].plot(x0,x1,color='black')

    plt.show()
