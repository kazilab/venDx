import pandas as pd
import numpy as np
import torch
import logging
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.io as pio
from collections import defaultdict
from .binary_tabnet import Classifier, TabPre
from .metrics import BinaryClassificationMetrics, BinaryClassificationLosses

# **** Build a TabNet model and calculate scores ****


def model_(data,
           test_data,
           data_labels,
           test_data_labels,
           test_size,
           unlabeled_data,
           bs,
           vbs,
           path,
           parameters,
           patience,
           max_epochs,
           random_state,
           sampling_method,
           param_search
           ):
    """
    Build a TabNet model and calculate scores.
    
    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        test_size: Test set size to split data.
        unlabeled_data: Unlabeled data for semi-supervised learning.
        bs: batch size
        vbs: virtual batch size.
        path: Path to store model and log files.
        parameters: TabNet model parameter's
        patience: Number of epochs with no improvement before early stopping.
        max_epochs: Maximum number of epochs to train for.
        random_state: Random state for reproducibility.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
    
    Returns:
        A tuple containing the unsupervised model, the supervised model, and a factor for performance.
    """
    logging.info('Building TabNet Model')
    print('Building TabNet Model')
    x_train_un, x_test_un = train_test_split(unlabeled_data.fillna(0).to_numpy(),
                                             test_size=float(test_size),
                                             random_state=int(random_state)
                                             )
    x_train, x_valid, y_train, y_valid = train_test_split(data.to_numpy(),
                                                          data_labels.to_numpy(),
                                                          test_size=float(test_size),
                                                          random_state=int(random_state)
                                                          )

    unsupervised_model = TabPre(
                                **parameters
                               )
    logging.info('Fitting unsupervised TabNet model')
    print('Fitting unsupervised TabNet model')
    unsupervised_model.fit(
                            X_train=x_train_un,
                            eval_set=[x_test_un],
                            batch_size=bs,
                            virtual_batch_size=vbs,
                            pretraining_ratio=0.8,
                            patience=patience,
                            max_epochs=max_epochs
                           )
    model = Classifier(**parameters
                       )
    logging.info('Fitting TabNet model')
    print('Fitting TabNet model')
    model.fit(X_train=x_train, y_train=y_train,
              eval_set=[(x_train, y_train), (x_valid, y_valid)],
              eval_name=['train', 'valid'],
              batch_size=bs,
              virtual_batch_size=vbs,
              eval_metric=["accuracy"],
              patience=patience,
              max_epochs=max_epochs,
              weights=1,
              from_unsupervised=unsupervised_model)
    
    filename = f"{path}{sampling_method}_{param_search}_TabNet_model"
    model.save_model(filename)
    logging.info("TabNet Model saved in venDx_result folder")
    plt.plot(model.history['loss'], 'b', label='Train loss')
    plt.plot(model.history['train_accuracy'], 'r', label='Train accuracy')
    plt.plot(model.history['valid_accuracy'], 'g', label='Validation accuracy')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    filename1 = f"{path}{sampling_method}_{param_search}_TabNet_valid_accuracy_curve.pdf"
    fig.savefig(filename1)
    plt.close()
    logging.info("Validation accuracy curve has been saved in venDx_result folder")
    del fig

    # Create a list of tuples containing score name, function and data type
    test_dict = defaultdict(dict)
    train_dict = defaultdict(dict)
    losses_dict = defaultdict(dict)
    
    test_instance = BinaryClassificationMetrics(model, test_data.to_numpy(), test_data_labels.to_numpy())
    train_instance = BinaryClassificationMetrics(model, x_train, y_train)
    loss_instance = BinaryClassificationLosses(model, test_data.to_numpy(), test_data_labels.to_numpy())
    test_sc = test_instance.fit()
    train_sc = train_instance.fit()
    losses = loss_instance.fit()
    
    for metric, score in test_sc.items():
        test_dict[metric] = score
    for metric, score in train_sc.items():
        train_dict[metric] = score
    for metric, score in losses.items():
        losses_dict[metric] = score
    
    test_df = pd.DataFrame.from_dict(test_dict, orient='index', columns=['Test'])
    train_df = pd.DataFrame.from_dict(train_dict, orient='index', columns=['Train'])
    losses_df = pd.DataFrame.from_dict(losses_dict, orient='index', columns=['Test loss'])
    drop_metrics = ["AUC", "Cohen's Kappa",	"F1 Score",	"Jaccard",	"MCC", "Average Precision Score"]

    # Calculate RMS of differences between train and test scores
    renamed_test_df = test_df.rename(columns=lambda col: col.replace("Test", "Diff")).drop(drop_metrics)
    renamed_train_df = train_df.rename(columns=lambda col: col.replace("Train", "Diff")).drop(drop_metrics)
    diff = np.mean(np.mean(np.square(renamed_train_df-renamed_test_df), axis=1))**0.5
    # Calculate RMS of test scores losses
    loss_ = np.mean(np.mean(np.square(losses_df.drop(drop_metrics)), axis=1))**0.5
    # Calculate final scores by taking geometric mean, and then making negative log2
    factor = -1*np.log2((diff*loss_)**0.5)
    all_score_df = pd.concat([train_df, test_df, losses_df], axis=1).round(decimals=3)
    all_score_df.loc['NegLog2-RMSS'] = factor  # Negative log2 Root Mean Squared Scores
    all_score_df.to_csv(f"{path}{sampling_method}_{param_search}_TabNet_scores.csv")
    logging.info("Scores have been saved in venDx_result folder in a csv file")
    return unsupervised_model, model, factor

# ****** ROC AUC Curve ******


def auc_roc(unsupervised_model,
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
            ):
    """
    Draw AUC-ROC curve using 5-fold cv.
    
    Args:
        unsupervised_model: model from unsupervised learning
        labeled_data_s: Labeled data after sampling
        labels_s: Labels after sampling
        parameters: TabNet model parameter's
        bs: Batch size.
        vbs: Virtual batch size.
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
    logging.info('Building RepeatedKFold TabNet Model')
    print('Building RepeatedKFold TabNet Model')
    kf = RepeatedKFold(n_splits=cv,
                       n_repeats=n_trials,
                       random_state=random_state
                       )
    metrics = ['auc', 'fpr', 'tpr', 'accuracy', 'thresholds']
    results = {m: [] for m in metrics}
    xx = labeled_data_s.values
    yy = labels_s.values
    for train, test in kf.split(xx):
        x_train, x_test = xx[train], xx[test]
        y_train, y_test = yy[train], yy[test]
        clf_roc = Classifier(**parameters
                             )
        clf_roc.fit(X_train=x_train, y_train=y_train,
                    eval_set=[(x_test, y_test)],
                    eval_name=['test'],
                    patience=patience,
                    max_epochs=max_epochs,
                    eval_metric=['accuracy'],
                    batch_size=bs,
                    virtual_batch_size=vbs,
                    weights=1,
                    from_unsupervised=unsupervised_model
                    )
        y_preds = clf_roc.predict(x_test)
        labels = y_test
        fpr, tpr, thresholds = roc_curve(labels, y_preds)
        results['fpr'].append(fpr)
        results['tpr'].append(tpr)
        results['thresholds'].append(thresholds)
        results['accuracy'].append(accuracy_score(labels, y_preds))
        results['auc'].append(roc_auc_score(labels, y_preds))
    
    pd.DataFrame(results).to_csv(f"{path}{sampling_method}_{param_search}_TabNet_repeated_accuracy.csv")
    logging.info("Repeated accuracy scores have been saved in venDx_result folder in a csv file")
    c_fill = 'rgba(52, 152, 219, 0.2)'
    c_line = 'rgba(52, 152, 219, 0.5)'
    c_line_main = 'rgba(41, 128, 185, 1.0)'
    c_grid = 'rgba(189, 195, 199, 0.5)'
    fpr_mean = np.linspace(0, 1, n_trials*cv)
    interp_tprs = []
    for i in range(n_trials*cv):
        fpr = results['fpr'][i]
        tpr = results['tpr'][i]
        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    tpr_mean = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std = 2*np.std(interp_tprs, axis=0)
    tpr_upper = np.clip(tpr_mean+tpr_std, 0, 1)
    tpr_lower = tpr_mean-tpr_std
    auc = np.mean(results['auc'])
    fig = go.Figure([
        go.Scatter(
            x=fpr_mean,
            y=tpr_upper,
            line=dict(color=c_line, width=1),
            hoverinfo="skip",
            showlegend=False,
            name='upper'),
        go.Scatter(
            x=fpr_mean,
            y=tpr_lower,
            fill='tonexty',
            fillcolor=c_fill,
            line=dict(color=c_line, width=1),
            hoverinfo="skip",
            showlegend=False,
            name='lower'),
        go.Scatter(
            x=fpr_mean,
            y=tpr_mean,
            line=dict(color=c_line_main, width=2),
            hoverinfo="skip",
            showlegend=True,
            name=f'AUC: {auc:.3f}')
    ])
    fig.add_shape(
        type='line',
        line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis_title="1 - Specificity",
        yaxis_title="Sensitivity",
        width=800,
        height=800,
        legend=dict(
            yanchor="bottom",
            xanchor="right",
            x=0.95,
            y=0.01,
        )
    )
    fig.update_yaxes(
        range=[0, 1],
        gridcolor=c_grid,
        scaleanchor="x",
        scaleratio=1,
        linecolor='black')
    fig.update_xaxes(
        range=[0, 1],
        gridcolor=c_grid,
        constrain='domain',
        linecolor='black')
    pio.write_image(fig, f"{path}{sampling_method}_{param_search}_TabNet_AUC_curve.pdf")
    logging.info("Repeated AUC curve has been saved in venDx_result folder")
    del fig, clf_roc, xx, yy, kf, metrics, labeled_data_s, labels_s
    
    # *** Draw accuracy plot ***
    y_accuracy = results['accuracy']
    av_accuracy = np.mean(y_accuracy)
    sem_accuracy = pd.DataFrame(y_accuracy).sem()
    trace = go.Scatter(
                        x=np.arange(1, n_trials*cv+1),  # Generate auto number for X axis
                        y=y_accuracy,
                        mode='markers',
                        marker=dict(color='lightblue')
                        )
    # Create the layout for the plot
    layout = go.Layout(
            title='Accuracy plot from cross-validation',
            xaxis=dict(title='Repeated cross-validation number',
                       showgrid=False,
                       range=[0, n_trials*cv+0.5],
                       showline=True, linewidth=0.1, linecolor='black',  # Add x-axis frame
                       mirror=True,  # Reflect ticks and frame on the opposite side
                       ),
            yaxis=dict(title='Accuracy',
                       showgrid=False,
                       # range=[0, 1],
                       showline=True, linewidth=0.1, linecolor='black',  # Add y-axis frame
                       mirror=True,  # Reflect ticks and frame on the opposite side
                       ),
            plot_bgcolor='rgba(0,0,0,0)',  # Set transparent background
            )
    # Create the figure and plot it
    fig1 = go.Figure(data=[trace], layout=layout)
    fig1.add_annotation(
                        x=1, y=0,
                        xref="paper", yref="paper",  # use relative coordinates
                        text=f"Average accuracy: {av_accuracy:.3f}, SEM: {sem_accuracy[0]:.5f}",
                        showarrow=False,
                        font=dict(size=10),
                        xanchor="right", yanchor="bottom",
                        # bgcolor="rgba(255,255,255,08)" # to add a white background
                        )
    pio.write_image(fig1, f"{path}{sampling_method}_{param_search}_accuracy_plot_k_fold_cv_roc.pdf")
    logging.info("Repeated accuracy plot has been saved in venDx_result folder")
    del results, fig1, y_accuracy, trace, layout
