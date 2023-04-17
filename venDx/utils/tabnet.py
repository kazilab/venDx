import pandas as pd
import logging
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .utils import sampling, batch_sizes, shap_
from .metrics import confusion_matrix_
from .optim import _optim
from .models import auc_roc
from .ml_models import run_ml_models

# ****************** Run model ******************************


def run_model(labeled_data,
              unlabeled_data,
              labels,
              path,
              test_size,
              patience,
              max_epochs,
              random_state,
              n_trials,
              cv,
              sampling_method,
              param_search,
              run_tabnet_model,
              run_feat_imp,
              run_shap,
              run_roc,
              run_ml
              ):
    """
    Build and use a TabNet model and calculate scores.
    
    Args:
        labeled_data: labeled data matrix.
        unlabeled_data: unlabeled data matrix .
        labels: Labels for data.
        path: Path to store model and log files.
        test_size: Test size to divide the data
        patience: Number of epochs with no improvement before early stopping.
        max_epochs: Maximum number of epochs to train for.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
        run_tabnet_model: whether TabNet model will be built
        run_feat_imp: Calculate feature importance
        run_shap: Calculate SHAP plots
        run_roc: Calculate AUC-ROC curve using 5-fold cv
        run_ml: Run ML models to compare.
    
    Returns:
        Save results in the venDx_Results folder.
    """
    logging.info('**** Building model with TabNet ****')
    print('**** Building model with TabNet ****')

    # Separate 20% test data for testing the model before using sampling methods
    train_data, test_data, train_labels, test_data_labels = train_test_split(labeled_data,
                                                                             labels,
                                                                             test_size=float(test_size),
                                                                             random_state=int(random_state)
                                                                             )
    # Use defined sampling methods, under sampling, overs sampling or no sampling.

    # Data for model optimization and building
    data, data_labels = sampling(train_data,
                                 train_labels,
                                 sampling_method
                                 )
    # Data for only repeated model building and auc_roc calculation
    labeled_data_s, labels_s = sampling(labeled_data,
                                        labels,
                                        sampling_method
                                        )

    if run_tabnet_model == True:
        # Calculate batch size and virtual batch size from train data
        bs, vbs = batch_sizes(
                              data
                              )
        # use _optim method to calculate parameters, and to build supervised and unsupervised models.
        if param_search == 'Best':
            parameters, unsupervised_model, model, name_ = _optim(data,
                                                                  test_data,
                                                                  data_labels,
                                                                  test_data_labels,
                                                                  test_size,
                                                                  unlabeled_data,
                                                                  path,
                                                                  patience,
                                                                  max_epochs,
                                                                  random_state,
                                                                  n_trials,
                                                                  cv,
                                                                  sampling_method,
                                                                  param_search
                                                                  )
            filename = f"{path}{sampling_method}_{param_search}_{name_}_TabNet_model"
            model.save_model(filename)
            logging.info("Best Tabnet model has been saved in venDx_results folder")
        else:
            parameters, unsupervised_model, model, _ = _optim(data,
                                                              test_data,
                                                              data_labels,
                                                              test_data_labels,
                                                              test_size,
                                                              unlabeled_data,
                                                              path,
                                                              patience,
                                                              max_epochs,
                                                              random_state,
                                                              n_trials,
                                                              cv,
                                                              sampling_method,
                                                              param_search
                                                              )
        params_df = pd.DataFrame.from_dict(parameters, orient='index', columns=['values']).T
        params_df['n_steps'] = params_df['n_steps'].convert_dtypes(False, False, True, False, False)
        params_df['n_d'] = params_df['n_d'].convert_dtypes(False, False, True, False, False)
        params_df['n_a'] = params_df['n_a'].convert_dtypes(False, False, True, False, False)
        params_df['n_independent'] = params_df['n_independent'].convert_dtypes(False, False, True, False, False)
        params_df['n_shared'] = params_df['n_shared'].convert_dtypes(False, False, True, False, False)
        params_df = params_df.T
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=params_df.values, rowLabels=params_df.index, loc='center')
        if param_search == 'Best':
            fig.savefig(f"{path}{sampling_method}_{param_search}_{name_}_parameters.pdf")
            logging.info("Best tabnet model parameters have been saved in venDx_results folder in a pdf file")
        else:
            fig.savefig(f"{path}{sampling_method}_{param_search}_parameters.pdf")
            logging.info("Ttabnet model parameters have been saved in venDx_results folder in a pdf file")
        plt.close()
        del params_df, fig
        
        if run_feat_imp == True:
            logging.info('Calculating feature importance')
            print('Calculating feature importance')
            feat_imp = pd.DataFrame(model.feature_importances_)
            perm_imp1 = permutation_importance(model,
                                               test_data.to_numpy(),
                                               test_data_labels.to_numpy(),
                                               scoring='accuracy'
                                               )
            perm_imp = pd.DataFrame(perm_imp1.importances_mean)
            combined = pd.concat([feat_imp, perm_imp], axis=1).T
            combined.columns = test_data.T.reset_index().iloc[:, 0].values.ravel()  # use gene name from the index
            combined_ = combined.T
            combined_.columns = ['Importance score', 'Permutation importance']
            combined_.index.names = ['Gene name']
            combined_.to_csv(f"{path}{sampling_method}_{param_search}_TabNet_global_feature_importance.csv")
            logging.info("TabNet global feature importance file has been saved in venDx_results folder")
            del feat_imp, perm_imp1, perm_imp, combined, combined_
        else:
            logging.info('Feature importance have not been calculated! ')
            logging.info('Please pass run_feat_imp=True to calculate feature importance.')
        
        # *** Drawing confusion matrix ***
        logging.info('Drawing confusion matrix')
        print('Drawing confusion matrix')
        confusion_matrix_(model,
                          test_data.to_numpy(),
                          test_data_labels.to_numpy(),
                          path,
                          sampling_method,
                          param_search
                          )
                          
        # *** Generating SHAP plots ***
        if run_shap == True:
            logging.info('Feature importance using SHAP')
            print('Feature importance using SHAP')
            shap_(model,
                  test_data,
                  path,
                  sampling_method,
                  param_search
                  )
        else:
            logging.info('SHAP plot has not been generated! Please pass run_shap=True to generate SHAP plot.')

        # *** Generating ROC plot ***
        if run_roc == True:
            logging.info('Drawing ROC curve')
            print('Drawing ROC curve')
            auc_roc(unsupervised_model,
                    labeled_data_s,
                    labels_s,
                    bs,
                    vbs,
                    path,
                    parameters,
                    patience,
                    max_epochs,
                    random_state,
                    n_trials,
                    cv,
                    sampling_method,
                    param_search
                    )
        else:
            logging.info('AUC-ROC curve has not been generated! Please pass run_roc=True to generate AUC-ROC curve.')
        del (labeled_data, unlabeled_data, labels, test_size, patience, max_epochs, n_trials,
             cv, run_feat_imp, run_shap, run_roc, bs, vbs, parameters, unsupervised_model, model,
             labeled_data_s, labels_s, run_tabnet_model)

    else:
        logging.info('We have not developed TabNet model! Please pass run_tabnet_model=True to generate TabNet model.')
    # *** Run machine learning models ***
    if run_ml == True:
        logging.info('Running machine learning models')
        print('Running machine learning models')
        run_ml_models(data,
                      test_data,
                      data_labels,
                      test_data_labels,
                      path,
                      random_state,
                      sampling_method,
                      param_search
                      )
    else:
        logging.info('Analysis with all ML algorithms has not been performed! '
                     'Please pass run_ml=True to perform analysis.')
    
    del data, test_data, data_labels, test_data_labels, path, random_state, sampling_method, param_search, run_ml
