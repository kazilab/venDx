import warnings
import os
import math
import shap
import logging
import scanpy as sc
import pandas as pd
import numpy as np
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------#
# data: Gene names in column header and sample names in index (row header)
# sample BCL2 BCL2L1 PLK1 ... ... ... ... ... ... ... ... ... ... ... FLT3
# spl1    12    39    22  ... ... ... ... ... ... ... ... ... ... ... 30
# spl2    10    22    13  ... ... ... ... ... ... ... ... ... ... ... 25
# -------------------------------------------------------------------------#

# file ext


def ext(data_path):
    _, data_file_ext = os.path.splitext(data_path)
    if data_file_ext == '.csv':
        sep = ','
    else:
        sep = '\t'
    return sep

# ******************** Highly Variable Genes *******************************
# Tasks                                                                    #
# 1. Take a pd.DataFrame                                                   #
# 2. Convert to sc.AnnData, check if data is log normalized                #
# 3. Calculate HVG using sc.pp.highly_variable_genes                       #
# 4. Export HVG to result folder                                           #
# --------------------------------------------------------------------------#


def hvg_(data, results_path, n_top_genes, min_disp, flavor):
    data = data.fillna(0)
    data = data.astype(np.float64)
    adata = sc.AnnData(data)
    if np.max(np.max(adata.X, axis=1)) >= 38:
        if np.min(np.min(adata.X, axis=1)) <= 1:
            adata.X = np.log2(adata.X + 1 - np.min(np.min(adata.X, axis=1)))
        else:
            adata.X = np.log2(adata.X)
    else:
        adata.X = adata.X
    sc.pp.highly_variable_genes(adata,
                                layer=None,
                                n_top_genes=n_top_genes,
                                min_disp=min_disp,
                                max_disp=np.inf,
                                min_mean=0.0125,
                                max_mean=3,
                                span=0.3,
                                n_bins=20,
                                flavor=flavor,
                                subset=False,
                                inplace=True,
                                batch_key=None,
                                check_values=True)
    hvg_genes = adata.var.index[adata.var['highly_variable']]
    df = pd.DataFrame(hvg_genes)
    df.columns = ['Highly Variable Genes']
    df.to_csv(results_path+'hvg.csv', index=False)
    logging.info("HVG has been saved in hvg.csv in venDx_results folder")
    return adata

# Min-Max normalization


def min_max(data, feature_min=0, feature_max=1):
    data = data
    data_min = np.nanmin(data, axis=0)
    data_max = np.nanmax(data, axis=0)
    x_std = (data - data_min) / (data_max - data_min)
    x_scaled = x_std*(feature_max - feature_min) + feature_min
    return x_scaled


# ***************** Data sampling ***********************************
# Tasks                                                              #
# 1. Take data and labels as pd.DataFrame                            #
# 2. Use imblearn to undersample or over sample                       #
# 3. Run under sampling in two steps                                  #
# 4. Run over sample using SMOTE                                      #
# --------------------------------------------------------------------#


def sampling(x_train, y_train, sampling_method):
    # type= under=under sampling, over=over sampling
    # applicable only for train data
    # do not sample test data
    sampling_method = str(sampling_method)
    if sampling_method == 'under':
        sample1 = TomekLinks()
        x1, y1 = sample1.fit_resample(x_train, y_train)
        sample = RandomUnderSampler()
        x, y = sample.fit_resample(x1, y1)
    elif sampling_method == 'over':
        sample = SMOTE()
        x, y = sample.fit_resample(x_train, y_train)
    elif sampling_method == 'no':
        x, y = x_train, y_train
    else:
        error_message = "Sampling type error!"
        logging.error(error_message)
        raise ValueError(error_message)
    return x, y


# Batch size and virtual batch size
# Calculate batch_size from training data


def batch_sizes(x):
    size_factor = 2**int(math.floor(math.log2(x.shape[0]*0.1)))
    # batch_size = bs, virtual_batch_size = vbs
    if size_factor >= 1024:
        bs = 1024
        vbs = 128
    elif size_factor >= 128:
        bs = size_factor
        vbs = size_factor/8
    elif size_factor >= 16:
        bs = size_factor
        vbs = size_factor/4
    else:
        bs, vbs = 16, 4
    return bs, vbs

#  SHAPLY


def shap_(model, x_test, path, sampling_method, param_search):
    explainer = shap.Explainer(model.predict, x_test.to_numpy())
    # shap_values = shap.TreeExplainer(model).shap_values(X_test)
    shap_values = explainer(x_test, max_evals=round(2*x_test.shape[1]+1))
    with PdfPages(f"{path}{sampling_method}_{param_search}_Global_X_TEST_SHAP.pdf") as pdf:
        shap.summary_plot(shap_values, x_test, plot_type='layered_violin', color='#cccccc', show=False)
        plt.title('SHAP layered violin plot gray scale')
        plt.grid(None)
        pdf.savefig(ransparent=False, facecolor='auto', edgecolor='auto')
        plt.close()
        shap.summary_plot(shap_values, x_test, plot_type='layered_violin', show=False)
        plt.title('SHAP layered violin plot')
        plt.grid(None)
        pdf.savefig(transparent=False, facecolor='w', edgecolor='auto')
        plt.close()
        shap.summary_plot(shap_values, x_test, plot_type='violin', show=False)
        plt.title('SHAP violin plot')
        plt.grid(None)
        pdf.savefig()
        plt.close()
        shap.summary_plot(shap_values, x_test, show=False)
        plt.title('SHAP dot plot')
        plt.grid(None)
        pdf.savefig()
        plt.close()
    logging.info("SHAP plots have been saved in a pdf file in venDx_results folder")
# scale positive weight


def scale_pos_weight_(y):
    spw = (len(y)-sum(y))/sum(y)
    return spw
