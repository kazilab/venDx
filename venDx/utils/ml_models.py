import pandas as pd
import numpy as np
import logging
from .utils import scale_pos_weight_
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC, SVC
from .metrics import BinaryClassificationMetrics, BinaryClassificationLosses
import warnings
from collections import defaultdict

# ***** Calculate AUC and Accuracy *****


def run_ml_models(x_train,
                  x_test,
                  y_train,
                  y_test,
                  path,
                  random_state,
                  sampling_method,
                  parameter_search
                  ):
    warnings.filterwarnings("ignore")
    
    # Define model parameters
    clf_sgdl2 = SGDClassifier(penalty='l1',
                              random_state=random_state,
                              loss='log_loss',
                              class_weight='balanced'
                              )
    clf_sgdl1 = SGDClassifier(penalty='l1',
                              loss='log_loss',
                              random_state=random_state,
                              class_weight='balanced'
                              )
    clf_sgden = SGDClassifier(penalty='elasticnet',
                              loss='log_loss',
                              random_state=random_state,
                              class_weight='balanced'
                              )
    clf_gnb = GaussianNB(
                        )
    clf_rfc = RandomForestClassifier(
                                     random_state=random_state,
                                     class_weight='balanced'
                                     )
    clf_cbc = CatBoostClassifier(verbose=False,
                                 random_state=random_state,
                                 auto_class_weights='Balanced'
                                 )
    clf_lgbm = LGBMClassifier(verbose=-1,
                              random_state=random_state,
                              class_weight='balanced'
                              )
    clf_xgbc = XGBClassifier(verbosity=0,
                             random_state=random_state,
                             scale_pos_weight=scale_pos_weight_(y_train)
                             )
    clf_knn = KNeighborsClassifier(
                                   )
    clf_mlpc = MLPClassifier(
                             random_state=random_state,
                             verbose=0
                             )
    clf_non_lin = NuSVC(random_state=18,
                        probability=True,
                        class_weight='balanced'
                        )
    clf_rbf = SVC(gamma='auto',
                  probability=True,
                  random_state=random_state,
                  class_weight='balanced'
                  )
    
    # Create a list of models
    clf_list = [
                (clf_sgdl2, "SGD L2"),
                (clf_sgdl1, "SGD L1"),
                (clf_sgden, "SGD Elastic Net"),
                (clf_gnb, "Gaussian NB"),
                (clf_rfc, "Random Forest"),
                (clf_cbc, "Cat Boost"),
                (clf_lgbm, "Light GBM"),
                (clf_xgbc, "XGBoostClassifier"),
                (clf_knn, "KNeighborsClassifier"),
                (clf_mlpc, "Multi-layer Perceptron"),
                (clf_non_lin, "Nu SVC"),
                (clf_rbf, "SVC rbf")
               ]
    
    test_dict = defaultdict(dict)
    train_dict = defaultdict(dict)
    losses_dict = defaultdict(dict)
    
    for clf, clf_name in clf_list:
        
        clf.fit(x_train, y_train)
        test_instance = BinaryClassificationMetrics(clf, x_test, y_test)
        train_instance = BinaryClassificationMetrics(clf, x_train, y_train)
        loss_instance = BinaryClassificationLosses(clf, x_test, y_test)
        test_sc = test_instance.fit()
        train_sc = train_instance.fit()
        losses = loss_instance.fit()
        
        for metric, score in test_sc.items():
            test_dict[clf_name]['Test '+metric] = score
        for metric, score in train_sc.items():
            train_dict[clf_name]['Train '+metric] = score
        for metric, score in losses.items():
            losses_dict[clf_name]['Test loss '+metric] = score
    
    test_df = pd.DataFrame(test_dict).T
    train_df = pd.DataFrame(train_dict).T
    losses_df = pd.DataFrame(losses_dict).T
    # Calculate RMS of differences between train and test scores
    drop_metrics = ["AUC", "Cohen's Kappa", "F1 Score", "Jaccard", "MCC", "Average Precision Score"]
    renamed_test_df = test_df.rename(columns=lambda col: col.replace("Test ", "")).drop(drop_metrics, axis=1)
    renamed_train_df = train_df.rename(columns=lambda col: col.replace("Train ", "")).drop(drop_metrics, axis=1)
    renamed_losses_df = losses_df.rename(columns=lambda col: col.replace("Test loss ", "")).drop(drop_metrics, axis=1)
    diff = np.mean(np.square(renamed_train_df-renamed_test_df), axis=1)**0.5
    # Calculate RMS of test scores losses
    loss_ = np.mean(np.square(renamed_losses_df), axis=1)**0.5
    # Calculate final scores by taking geometric mean, and then making negative log2
    final_scores = pd.DataFrame(-1*np.log2((diff*loss_)**0.5))
    final_scores.columns = ['NegLog2-RMSS']  # Negative log2 Root Mean Squared Scores
        
    all_score_df = pd.concat([train_df, test_df, losses_df, final_scores], axis=1).round(decimals=3)
    # test_df.to_csv(f'{path}{sampling_method}_{parameter_search}_Train_Test_scores_for_all_ML_test.csv')
    # train_df.to_csv(f'{path}{sampling_method}_{parameter_search}_Train_Test_scores_for_all_ML_train.csv')
    # losses_df.to_csv(f'{path}{sampling_method}_{parameter_search}_Train_Test_losses_for_all_ML.csv')
    # final_scores.to_csv(f'{path}{sampling_method}_{parameter_search}_Train_Test_final_scores_for_all_ML.csv')
    all_score_df.to_csv(f'{path}{sampling_method}_{parameter_search}_Train_Test_all_scores_for_all_ML.csv')
    logging.info("Scores have been saved in a csv file in venDx_result folder")
