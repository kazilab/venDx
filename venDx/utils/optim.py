import numpy as np
import optuna
import logging
from optuna import visualization
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
from .metrics import custom_score, combined_kappa_mcc
from .utils import batch_sizes
from .models import model_
from ..arguments import arguments
from .binary_tabnet import Classifier
from sklearn.metrics import make_scorer
# ***** Parameter optimization *****
# ***** Parameter search by OPTUNA *****


def _optuna(data,
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
            ):
    """
    Perform Optuna hyperparameter search for a given model.
    
    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        test_size: Test set size to split data.
        unlabeled_data: Unlabeled data for semi-supervised learning.
        path: Path to store model and log files.
        patience: Number of epochs with no improvement before early stopping.
        max_epochs: Maximum number of epochs to train for.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
    
    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    logging.info('Searching parameter using Optuna')
    print('Searching parameter using Optuna')

    def objective(trial):
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_d = trial.suggest_int("n_d", 8, 64, step=1)
        n_steps = trial.suggest_int("n_steps", 3, 5, step=1)
        gamma = trial.suggest_float("gamma", 1., 1.5, step=0.1)
        lambda_sparse = trial.suggest_float("lambda_sparse", 0.00001, 0.0001, log=True)

        params_ = dict(n_d=n_d,
                       n_a=n_d,
                       n_steps=n_steps,
                       gamma=gamma,
                       n_independent=3,
                       n_shared=3,
                       mask_type=mask_type,
                       lambda_sparse=lambda_sparse,
                       seed=random_state
                       )
        kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)
        cv_score_array = []
        xx = data.to_numpy()
        yy = data_labels.to_numpy()
        for train_index, test_index in kf.split(xx):
            x_train, x_valid = xx[train_index], xx[test_index]
            y_train, y_valid = yy[train_index], yy[test_index]
            clf_opti = Classifier(**params_)
            clf_opti.fit(X_train=x_train, y_train=y_train,
                         eval_set=[(x_valid, y_valid)],
                         eval_name=['valid'],
                         patience=patience,
                         max_epochs=max_epochs,
                         eval_metric=['cohen_mcc'],
                         weights=1,
                         batch_size=batch_sizes(x_train)[0],
                         virtual_batch_size=batch_sizes(x_train)[1]
                         )
            cs = custom_score(clf_opti, x_train, y_train, x_valid, y_valid)
            cv_score_array.append(cs)
            # print('cv_score: ', cs)
        avg = np.mean(cv_score_array)
        # print('Average cv_score: ', avg)
        return avg

    study = optuna.create_study(direction="minimize",
                                study_name='Optuna optimization',
                                sampler=TPESampler()
                                )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, timeout=36000)
    fig = visualization.plot_param_importances(study)
    filename = f"{path}{sampling_method}_{param_search}_TabNet_optuna_param_importance.pdf"
    fig.write_image(filename)
    logging.info("Optuna parameter importance has been saved in venDx_result folder")
    
    additional_parameters = {
                            'n_a': study.best_params['n_d'],
                            'n_independent': 3,
                            'n_shared': 3,
                            'seed': random_state
                            }
    parameters = study.best_params
    parameters.update(additional_parameters)

    unsupervised_model, model, performance = model_(data,
                                                    test_data,
                                                    data_labels,
                                                    test_data_labels,
                                                    test_size,
                                                    unlabeled_data,
                                                    batch_sizes(data)[0],
                                                    batch_sizes(data)[1],
                                                    path,
                                                    parameters,
                                                    patience,
                                                    max_epochs,
                                                    random_state,
                                                    sampling_method,
                                                    param_search
                                                    )
    return parameters, unsupervised_model, model, performance

# ***** BayesSearchCV *****


def _bayes(data,
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
           ):
    """
    Perform Bayesian optimization using BayesSearchCV for a given model.
    
    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        test_size: Test set size to split data.
        unlabeled_data: Unlabeled data for semi-supervised learning.
        path: Path to store model and log files.
        patience: Number of epochs with no improvement before early stopping.
        max_epochs: Maximum number of epochs to train for.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
    
    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    logging.info('Searching parameter using BayesSearchCV')
    print('Searching parameter using BayesSearchCV')
    x_train, x_valid, y_train, y_valid = train_test_split(data.to_numpy(),
                                                          data_labels.to_numpy(),
                                                          test_size=float(test_size),
                                                          random_state=int(random_state)
                                                          )
    custom_scorer = make_scorer(combined_kappa_mcc, greater_is_better=False)

    space = {'mask_type': Categorical(["entmax", "sparsemax"]),
             'n_d': Integer(8, 64),
             'n_a': Integer(8, 64),
             'n_steps': Integer(3, 5),
             'gamma': Real(1.1, 1.5),
             'lambda_sparse': Real(0.00001, 0.0001, 'log-uniform'),
             }
    opt = BayesSearchCV(Classifier(n_independent=3,
                                   n_shared=3,
                                   seed=random_state
                                   ),
                        search_spaces=space,
                        scoring=custom_scorer,  # 'neg_log_loss',
                        n_iter=n_trials,
                        cv=cv,
                        verbose=0,
                        random_state=random_state
                        )
    opt.fit(x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_name=['valid'],
            batch_size=batch_sizes(x_train)[0],
            virtual_batch_size=batch_sizes(x_train)[1],
            patience=patience,
            max_epochs=max_epochs,
            eval_metric=['cohen_mcc'],
            weights=1
            )
    
    additional_parameters = {
                            'n_independent': 3,
                            'n_shared': 3,
                            'seed': random_state
                            }
    parameters = opt.best_params_
    parameters.update(additional_parameters)

    unsupervised_model, model, performance = model_(data,
                                                    test_data,
                                                    data_labels,
                                                    test_data_labels,
                                                    test_size,
                                                    unlabeled_data,
                                                    batch_sizes(data)[0],
                                                    batch_sizes(data)[1],
                                                    path,
                                                    parameters,
                                                    patience,
                                                    max_epochs,
                                                    random_state,
                                                    sampling_method,
                                                    param_search
                                                    )
    return parameters, unsupervised_model, model, performance

# ****** GridSearchCV *******


def _grid(data,
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
          ):
    """
    Perform optimization using GridSearchCV for a given model.
    
    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        test_size: Test set size to split data.
        unlabeled_data: Unlabeled data for semi-supervised learning.
        path: Path to store model and log files.
        patience: Number of epochs with no improvement before early stopping.
        max_epochs: Maximum number of epochs to train for.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
    
    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    logging.info('Searching parameter using GridSearchCV')
    print('Searching parameter using GridSearchCV')
    x_train, x_valid, y_train, y_valid = train_test_split(data.to_numpy(),
                                                          data_labels.to_numpy(),
                                                          test_size=float(test_size),
                                                          random_state=int(random_state)
                                                          )
    custom_scorer = make_scorer(combined_kappa_mcc, greater_is_better=False)
    space = {'mask_type': ["entmax", "sparsemax"],
             'n_d': [22],
             # 'n_d': np.linspace(8,64,28, endpoint=True, dtype=int).tolist(),
             'n_a': [22],
             # 'n_a': np.linspace(8,64,28, endpoint=True, dtype=int).tolist(),
             'n_steps': np.linspace(3, 5, 3, endpoint=True, dtype=int).tolist(),
             'gamma': [1.2],
             # 'gamma': np.linspace(1.1,1.2,2, endpoint=True, dtype=float).tolist(),
             'lambda_sparse': np.logspace(-5, -4, num=5, endpoint=True, base=10.0, dtype=float).tolist(),
             }

    opt = GridSearchCV(Classifier(n_independent=3,
                                  n_shared=3,
                                  seed=random_state,
                                  ),
                       param_grid=space,
                       scoring=custom_scorer,  # 'neg_log_loss',
                       cv=cv,
                       verbose=0
                       )
    opt.fit(x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_name=['valid'],
            batch_size=batch_sizes(x_train)[0],
            virtual_batch_size=batch_sizes(x_train)[1],
            patience=patience,
            max_epochs=max_epochs,
            eval_metric=['cohen_mcc'],
            weights=1
            )
    
    additional_parameters = {
                            'n_independent': 3,
                            'n_shared': 3,
                            'seed': random_state
                            }
    parameters = opt.best_params_
    parameters.update(additional_parameters)

    unsupervised_model, model, performance = model_(data,
                                                    test_data,
                                                    data_labels,
                                                    test_data_labels,
                                                    test_size,
                                                    unlabeled_data,
                                                    batch_sizes(data)[0],
                                                    batch_sizes(data)[1],
                                                    path,
                                                    parameters,
                                                    patience,
                                                    max_epochs,
                                                    random_state,
                                                    sampling_method,
                                                    param_search
                                                    )
    return parameters, unsupervised_model, model, performance

# ***** Predefined parameter *****


def _predefined(data,
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
                ):
    """
    Build a model using Predefined parameters.
    
    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        test_size: Test set size to split data.
        unlabeled_data: Unlabeled data for semi-supervised learning.
        path: Path to store model and log files.
        patience: Number of epochs with no improvement before early stopping.
        max_epochs: Maximum number of epochs to train for.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
    
    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    # Hardcoded Predefined parameters
    # These parameters can be obtained from prior experiments, domain knowledge, or literature
    logging.info('Model will be built using Predefined parameters')
    print('Model will be built using Predefined parameters')
    args = arguments()
    parameters = {'mask_type': args.mask_type,
                  'n_d': args.n_d_a,
                  'n_a': args.n_d_a,
                  'n_steps': args.n_steps,
                  'gamma': args.gamma,
                  'n_independent': 3,
                  'n_shared': 3,
                  'lambda_sparse': args.lambda_sparse,
                  'seed': random_state
                  }
    unsupervised_model, model, performance = model_(data,
                                                    test_data,
                                                    data_labels,
                                                    test_data_labels,
                                                    test_size,
                                                    unlabeled_data,
                                                    batch_sizes(data)[0],
                                                    batch_sizes(data)[1],
                                                    path,
                                                    parameters,
                                                    patience,
                                                    max_epochs,
                                                    random_state,
                                                    sampling_method,
                                                    param_search
                                                    )
    return parameters, unsupervised_model, model, performance

# Search for the best parameters
# Here we use all four methods for parameter search
# Develop TabNet model using each search parameters
# Calculate scores to check the quality of the model
# Calculate combined score: RMS of Test scores (if scores value is close to 1, then 1-score, or the score value)
# Calculate combined difference: RMS of Train score-Test score
# Geometric mean of Calculate combined score and Calculate combined difference
# Find the minimum geometric mean for the best parameter selection


def best_search(data,
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
                ):
    """
    Find the best search method and its corresponding parameters, unsupervised_model, model, and performance.

    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        test_size: Test set size to split data.
        unlabeled_data: Unlabeled data.
        path: Path to save model.
        patience: Patience for early stopping.
        max_epochs: Maximum number of epochs.
        random_state: Seed for random number generator.
        n_trials: Number of trials.
        cv: Cross-validation strategy.
        sampling_method: Method for sampling.
        param_search: Parameters for the search space.

    Returns:
        Tuple containing the best method's parameters, unsupervised_model, model, and performance.
    """
    logging.info('Searching parameters using Best search')
    print('Searching parameters using Best search')

    def _run_method(method, *args, **kwargs):
        """
        Run the given method and return its results.

        Args:
            method: Function to run.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            Dictionary containing 'parameters', 'unsupervised_model', 'model', and 'performance'.
        """
        parameters, unsupervised_model, model, performance = method(*args, **kwargs)
        return {'parameters': parameters, 'unsupervised_model': unsupervised_model,
                'model': model, 'performance': performance}

    methods = [(_optuna, 'Optuna'), (_bayes, 'Bayes'), (_grid, 'Grid'), (_predefined, 'Predefined')]
    results = {name: _run_method(method,
                                 data,
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
                                 param_search+name
                                 ) for method, name in methods
               }

    # Determine which method produced the best results
    best_method = max(results, key=lambda name: results[name]['performance'])

    # Return the results of the best method
    best_parameters = results[best_method]['parameters']
    unsupervised_model_ = results[best_method]['unsupervised_model']
    best_model = results[best_method]['model']
    # best_score = results[best_method]['performance']
    return best_parameters, unsupervised_model_, best_model, best_method

# Use the selected method to search parameters


def _optim(data,
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
           ):
    """
    Trains a model using a specified hyperparameter search method.

    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        test_size: Test set size to split data.
        unlabeled_data: Unlabeled data for semi-supervised learning.
        path: Path to store model and log files.
        patience: Number of epochs with no improvement before early stopping.
        max_epochs: Maximum number of epochs to train for.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.

    Returns:
        A dictionary containing the best hyperparameter, the unsupervised model,
        the supervised model, and the training history.
    """
    logging.info('Running hyperparameter optimization')
    print('Running hyperparameter optimization')
    search_methods = {
        'Best': best_search,
        'Optuna': _optuna,
        'Bayes': _bayes,
        'Grid': _grid,
        'Predefined': _predefined
    }
    
    if param_search not in search_methods:
        error_message = "Please suggest a search parameter from the Best, Optuna, Bayes, Grid, and Predefined"
        logging.error(error_message)
        raise ValueError(error_message)
    
    search_method = search_methods[param_search]

    parameters, unsupervised_model, model, performance = search_method(data,
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
    
    return parameters, unsupervised_model, model, performance
