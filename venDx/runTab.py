import warnings
import time
import os
import logging
import pandas as pd
from .arguments import arguments
from .utils import ext, hvg_, min_max, run_model

# --------------------------------------------------------------------------*
# data: Gene names in column header and sample names in index (row header)  *
# sample BCL2 BCL2L1 PLK1 ... ... ... ... ... ... ... ... ... ... ... FLT3  *
# spl1    12    39    22  ... ... ... ... ... ... ... ... ... ... ... 30    *
# spl2    10    22    13  ... ... ... ... ... ... ... ... ... ... ... 25    *
# --------------------------------------------------------------------------*

# *************************** Note *****************************************
# Tasks                                                                    *
# sig1=gene_signature_1, sig2=gene_signature_2, hvg                        *
# sampling_type: under=under sampling, over=over sampling                  *
# -------------------------------------------------------------------------*


def run_tabnet(sampling_method="no",
               param_search="Predefined",
               run_tabnet_model=True,
               run_feat_imp=False,
               run_shap=False,
               run_roc=False,
               run_ml=False
               ):
    """
    sampling_method:
                     no: no sampling will be applied, data will be used as is.
                  under: training data will be under sampled using imblearn
                   over: training data will be oversampled using imblearn
                    all: all three sampling methods will be applied on training data
    param_search:
                Optuna: optuna will be used for parameter search
                 Bayes: BayesSearchCV will be used for parameter search
                  Grid: GridSearchCV will be used for parameter search
            Predefined: Predefined parameters will be used for parameter search
                  Best: All four parameter search will be performed and best parameters will be used
    runFeatImp:
                  True: Feature importance will be calculated using sklearn permutation_importance
                 False: To skip
    runSHAP:
                  True: SHAP plot will be generated
                 False: Skip
    runROC:
                  True: AUC-ROC curve will be generated using cross-validation
                 False: Skip
    runML:
                  True: A panel of machine learning algorithms will be used for comparison, scores will be generated
                 False: Skip
    """
    # Create results folder if needed
    if not os.path.exists('./venDx_results'):
        os.makedirs('./venDx_results')
    if not os.path.exists('./venDx_log'):
        os.makedirs('./venDx_log')
    result_path = './venDx_results/'
    logging_path = './venDx_log/'

    # Set the log level (DEBUG, INFO, WARNING, ERROR, or CRITICAL)
    log_level = logging.INFO

    # Set the log file path
    log_file_path = logging_path + "venDx_run.log"

    # Configure logging settings
    logging.basicConfig(filename=log_file_path,
                        level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Sampling methods
    sampling_methods = ['no', 'over', 'under', 'all']
    if sampling_method not in sampling_methods:
        raise ValueError(f"Invalid sampling method '{sampling_method}'."
                         f" Available options are: {', '.join(sampling_methods[:-1])}, {sampling_methods[-1]}")
    # Parameter search methods
    param_search_methods = ['Optuna', 'Bayes', 'Grid', 'Predefined', 'Best']
    if param_search not in param_search_methods:
        raise ValueError(f"Invalid parameter search method '{param_search}'. "
                         f"Available options are: {', '.join(param_search_methods[:-1])}, {param_search_methods[-1]}")
    
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    warnings.filterwarnings("ignore")
    logging.info('*** Analysis started ***')
    logging.info('Start time : %s', start)
    print('*** Analysis started *** @ ', start)

    # Extract parameters from arguments
    logging.info('Extracting parameters')
    args = arguments()
    logging.info('Loading data')
    data_labels = pd.read_csv(args.data_labels,
                              index_col=0,
                              header=0,
                              sep=ext(args.data_labels),
                              low_memory=False
                              )
    labeled_data = pd.read_csv(args.labeled_data,
                               index_col=0,
                               header=0,
                               sep=ext(args.labeled_data),
                               low_memory=False
                               )
    unlabeled_data = pd.read_csv(args.unlabeled_data,
                                 index_col=0,
                                 header=0,
                                 sep=ext(args.unlabeled_data),
                                 low_memory=False
                                 )
    data_for_hvg = pd.read_csv(args.data_for_hvg,
                               index_col=0,
                               header=0,
                               sep=ext(args.data_for_hvg),
                               low_memory=False
                               )

    logging.info('Searching for highly variable genes')
    adata = hvg_(data=data_for_hvg,
                 results_path=result_path,
                 n_top_genes=args.n_top_genes,
                 min_disp=args.min_disp,
                 flavor=args.flavor
                 )
    hvgs = adata.var.index[adata.var['highly_variable']]
    
    # Select data using features from HVG analysis
    sel_labeled_data = labeled_data[hvgs]
    sel_unlabeled_data = unlabeled_data[hvgs]
    
    # Normalize data using min-max normalization between 0 and 1.
    sel_labeled_data_mm = min_max(
                                  sel_labeled_data
                                  )
    sel_unlabeled_data_mm = min_max(
                                    sel_unlabeled_data
                                    )
    
    # Encode labels "Change here drug name if you have a table with multiple drugs
    labels = data_labels['Sensitivity'].replace(['resistant', 'sensitive'], [1, 0])

    # Run model depending on sampling method
    if sampling_method == 'all':
        for method in ['no', 'over', 'under']:
            logging.info(f'Using {method} sampling method for this analysis!')
            print(f'Using {method} sampling method for this analysis!')
            run_model(
                    labeled_data=sel_labeled_data_mm,
                    unlabeled_data=sel_unlabeled_data_mm,
                    labels=labels,
                    path=result_path,
                    test_size=args.test_size,
                    patience=args.patience,
                    max_epochs=args.max_epochs,
                    random_state=args.random_state,
                    n_trials=args.n_trials,
                    cv=args.cv,
                    sampling_method=method,
                    param_search=param_search,
                    run_tabnet_model=run_tabnet_model,
                    run_feat_imp=run_feat_imp,
                    run_shap=run_shap,
                    run_roc=run_roc,
                    run_ml=run_ml
                    )
            print(f'\n***###***###***\n')
    else:
        logging.info(f'Using {sampling_method} sampling method for this analysis!')
        print(f'Using {sampling_method} sampling method for this analysis!')
        run_model(
                labeled_data=sel_labeled_data_mm,
                unlabeled_data=sel_unlabeled_data_mm,
                labels=labels,
                path=result_path,
                test_size=args.test_size,
                patience=args.patience,
                max_epochs=args.max_epochs,
                random_state=args.random_state,
                n_trials=args.n_trials,
                cv=args.cv,
                sampling_method=sampling_method,
                param_search=param_search,
                run_tabnet_model=run_tabnet_model,
                run_feat_imp=run_feat_imp,
                run_shap=run_shap,
                run_roc=run_roc,
                run_ml=run_ml
                )
    del (sampling_methods, param_search_methods, result_path, data_labels, labeled_data, unlabeled_data, data_for_hvg,
         adata, hvgs, sel_labeled_data, sel_unlabeled_data, sel_labeled_data_mm, sel_unlabeled_data_mm, labels,
         sampling_method, param_search)
    logging.info('**** Analysis finished ****')
    finished = time.strftime("%Y-%m-%d %H:%M:%S")
    logging.info('Finish time : %s', finished)
    print('\n **** Analysis finished **** @ ', finished)
