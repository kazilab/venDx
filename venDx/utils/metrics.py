import warnings
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, jaccard_score, fbeta_score
from sklearn.metrics import hamming_loss, zero_one_loss
from sklearn.metrics import log_loss, brier_score_loss, average_precision_score, hinge_loss
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Metrics


class BinaryClassificationMetrics:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_true = y_test
        self.y_pred = model.predict(x_test)
        self.y_pred_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
        self.cm = confusion_matrix(self.y_true, self.y_pred)
        self.tn, self.fp, self.fn, self.tp = self.cm.ravel()
    
    def accuracy_test(self):  # 1 Accuracy
        return (self.tp+self.tn)/(self.tp+self.fn+self.fp+self.tn)
        
    def hamming_loss_test(self):  # 1-1 Hamming loss, same as zero_one_loss, and (1-accuracy) for binary classification
        return hamming_loss(self.y_true, self.y_pred)
    
    def zero_one_loss_test(self):  # 1-2 similar to hamming_loss, and (1-accuracy) for binary classification
        return zero_one_loss(self.y_true, self.y_pred)
    
    def precision_test(self):  # 2 Precision or Positive predictive value (PPV)
        return self.tp/(self.tp+self.fp)
        
    def fdr_test(self):  # 2-1 False discovery rate (FDR) is equal to (1-precission)
        return self.fp/(self.fp+self.tp)
    
    def sensitivity_test(self):  # 3 Sensitivity or Recall or True positive rate (TPR) or Detection Rate (DR)
        return self.tp/(self.tp+self.fn)

    def fnr_test(self):  # 3-1 False negative rate (FNR), False omission rate (FOR), equal to (1-sensitivity)
        return self.fn/(self.fn+self.tp)
        
    def specificity_test(self):  # 4 Specificity
        return self.tn/(self.tn+self.fp)
        
    def fpr_test(self):  # 4-1 False positive rate (FPR), False alarm rate, FAR, equal to (1-specificty)
        return self.fp/(self.fp+self.tn)
    
    def npv_test(self):  # 5 Negative predictive value (NPV)
        return self.tn/(self.fn+self.tn)

    def for_test(self):  # 5-1 False Omission Rate (FOR), equal to (1-NPV)
        return self.fp/(self.fp+self.tn)

    def f1_score_test(self):  # 6 F1, Harmonic mean of precision and recall
        return 2*self.tp/(2*self.tp+self.fp+self.fn)
    
    def fbeta_score_test(self):  # 6-1 F-beta score sklearn when beta=1, it is f1 score
        return fbeta_score(self.y_true, self.y_pred, beta=0.5)
    
    def balanced_accuracy_test(self):  # 7 Balanced accuracy
        return 0.5*(self.tp/(self.tp+self.fn)+self.tn/(self.fp+self.tn))
    
    def mcc_test(self):  # 8 Matthews correlation coefficient
        t = self.tp*self.tn-self.fp*self.fn
        b = (self.tp+self.tn)*(self.tp+self.fn)*(self.tn+self.fp)*(self.tn+self.fn)
        return t/(b**0.5)
    
    def cohen_kappa_test(self):  # 9 Cohen's Kappa, see sklearn.metrics.cohen_kappa_score, formula from wikipedia
        t = self.tp*self.tn-self.fp*self.fn
        b = (self.tp+self.fp)*(self.tn+self.fp)+(self.tp+self.fn)*(self.tn+self.fn)
        return 2*t/b
    
    def jaccard_test(self):  # 10 Jaccard
        return jaccard_score(self.y_true, self.y_pred)
    
    def negative_likelihood_ratio_test(self):  # 11 sklearn.metrics.class_likelihood_ratios (FNR/TNR)
        return (1-self.sensitivity_test())/self.specificity_test()
    
    def auc_proba_test(self):  # 12 ROC-AUC score from probability
        return roc_auc_score(self.y_true, self.y_pred_proba)
    
    def auc_test(self):  # 12-1 ROC-AUC score from probability
        return roc_auc_score(self.y_true, self.y_pred)
    
    def log_loss_test(self):  # 13 log loss from probability
        return log_loss(self.y_true, self.y_pred_proba)
    
    def brier_score_loss_test(self):  # 14 Brier score loss from probability
        return brier_score_loss(self.y_true, self.y_pred_proba)
    
    def average_precision_score_test(self):  # 15 Average Precision Score from probability
        return average_precision_score(self.y_true, self.y_pred_proba)
    
    def hinge_loss_test(self):  # 16 Hinge loss
        return hinge_loss(self.y_true, self.y_pred_proba)

    def fit(self):
        metrics = {
            "Accuracy": self.accuracy_test(),
            "AUC": self.auc_test(),
            "Cohen's Kappa": self.cohen_kappa_test(),
            "F1 Score": self.f1_score_test(),
            "Jaccard": self.jaccard_test(),
            "MCC": self.mcc_test(),
            "Negative Likelihood Ratio": self.negative_likelihood_ratio_test(),
            "NPV": self.npv_test(),
            "Precision": self.precision_test(),
            "Sensitivity": self.sensitivity_test(),
            "Specificity": self.specificity_test()
        }
        
        if self.y_pred_proba is not None:
            metrics.update({
                "Average Precision Score": self.average_precision_score_test(),
                "Brier Score Loss": self.brier_score_loss_test()
            })

        return metrics

# Metrics


class BinaryClassificationLosses:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_true = y_test
        self.y_pred = model.predict(x_test)
        self.y_pred_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
        self.cm = confusion_matrix(self.y_true, self.y_pred)
        self.tn, self.fp, self.fn, self.tp = self.cm.ravel()
    
    def accuracy_loss(self):  # 1 Accuracy
        return 1-(self.tp+self.tn)/(self.tp+self.fn+self.fp+self.tn)
    
    def precision_loss(self):  # 2 Precision or Positive predictive value (PPV) loss or FDR
        return 1-self.tp/(self.tp+self.fp)
    
    def sensitivity_loss(self):  # 3 Sensitivity or Recall or True positive rate (TPR) or Detection Rate (DR)
        return 1-self.tp/(self.tp+self.fn)
    
    def specificity_loss(self):  # 4 Specificity
        return 1-self.tn/(self.tn+self.fp)
    
    def npv_loss(self):  # 5 Negative predictive value (NPV)
        return 1-self.tn/(self.fn+self.tn)
    
    def f1_score_loss(self):  # 7 F1, Harmonic mean of precision and recall
        return 1-2*self.tp/(2*self.tp+self.fp+self.fn)
    
    def balanced_accuracy_loss(self):  # 8 Balanced accuracy similar to 0.5(FNR+FPR)
        return 1-0.5*(self.tp/(self.tp+self.fn)+self.tn/(self.fp+self.tn))
    
    def mcc_loss(self):  # 11 Matthews correlation coefficient
        t = self.tp*self.tn-self.fp*self.fn
        b = (self.tp+self.tn)*(self.tp+self.fn)*(self.tn+self.fp)*(self.tn+self.fn)
        return 1-t/(b**0.5)
    
    def cohen_kappa_loss(self):  # 12 Cohen's Kappa, see sklearn.metrics.cohen_kappa_score, formula from wikipedia
        t = self.tp*self.tn-self.fp*self.fn
        b = (self.tp+self.fp)*(self.tn+self.fp)+(self.tp+self.fn)*(self.tn+self.fn)
        return 1-2*t/b
    
    def jaccard_loss(self):  # 13 Jaccard
        return 1-jaccard_score(self.y_true, self.y_pred)
    
    def negative_likelihood_ratio_test(self):  # sklearn.metrics.class_likelihood_ratios (FNR/TNR)
        return (1-(self.tp/(self.tp+self.fn)))/(self.tn/(self.tn+self.fp))
    
    def fbeta_score_loss(self):  # 15 F-beta score sklearn
        return 1-fbeta_score(self.y_true, self.y_pred, beta=0.5)
    
    def auc_loss(self):  # 18 ROC-AUC score not from probability
        return 1-roc_auc_score(self.y_true, self.y_pred)
    
    def log_loss_test(self):  # 19 log loss from probability
        return log_loss(self.y_true, self.y_pred_proba)
    
    def brier_score_loss_test(self):  # 20 Brier score loss from probability
        return brier_score_loss(self.y_true, self.y_pred_proba)
    
    def average_precision_score_loss(self):  # 21 Average Precision Score from probability
        return 1-average_precision_score(self.y_true, self.y_pred_proba)
    
    def hinge_loss_test(self):  # 22 Hinge loss
        return hinge_loss(self.y_true, self.y_pred_proba)

    def fit(self):
        metrics = {
            "Accuracy": self.accuracy_loss(),  # Accuracy/Hamming/Zero-one loss
            "AUC": self.auc_loss(),
            "Cohen's Kappa": self.cohen_kappa_loss(),
            "F1 Score": self.f1_score_loss(),
            "Jaccard": self.jaccard_loss(),
            "MCC": self.mcc_loss(),
            "Negative Likelihood Ratio": self.negative_likelihood_ratio_test(),
            "NPV": self.npv_loss(),  # NPV loss (FOR)
            "Precision": self.precision_loss(),  # Precision loss (FDR)
            "Sensitivity": self.sensitivity_loss(),  # Sensitivity loss (FNR)
            "Specificity": self.specificity_loss()  # Specificity loss (FPR)
        }
        
        if self.y_pred_proba is not None:
            metrics.update({
                "Average Precision Score": self.average_precision_score_loss(),
                "Brier Score Loss": self.brier_score_loss_test()
            })

        return metrics

# Make a custom scorer to use for optimization


def combined_kappa_mcc(y_true, y_pred):
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return 1-(kappa*mcc)**0.5


def custom_score(model, x_train, y_train, x_test, y_test):
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_score = combined_kappa_mcc(y_train, train_pred)
    test_score = combined_kappa_mcc(y_test, test_pred)
    diff = (train_score - test_score)**2
    ts = test_score**2
    return (diff*ts)**0.25

# ***************** Confusion Matrix by SKLEARN **********************************


def confusion_matrix_(model, x_test, y_true, _path, sampling_method, param_search):
    # *******************************************
    #                        Predicted          *
    #                   Negative Positive       *
    # Actual Negative       TN      FP          *
    #       Positive        FN      TP          *
    # *******************************************

    y_pred_ = pd.DataFrame(model.predict(x_test)).values
    y_true_ = pd.DataFrame(y_true).values
    plt.rcParams['font.size'] = '12'

    cm_ = confusion_matrix(y_true_, y_pred_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_,
                                  display_labels=(['Sensitive', 'Resistant']))
    fig, ax = plt.subplots(figsize=(6, 4))
    disp.plot(ax=ax)
    ax.set_title('Confusion matrix', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label, (R: ' + str(sum(y_true)) + ', S: ' + str(len(y_true) - sum(y_true)) + ')', fontsize=12)
    fig.savefig(_path+sampling_method+'_'+param_search+'_TabNet_confusion_matrix.pdf')
    plt.close()
    del fig, cm_
