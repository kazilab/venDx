import argparse
import venDx
import os
import warnings

# store input variables
def arguments():
    parser = argparse.ArgumentParser(description='venDx')

    package_path = os.path.dirname(venDx.__file__)
    data_folder_path = os.path.join(package_path, "data")
    # data
    parser.add_argument('--unlabeled_data', type=str,
                        default=os.path.join(data_folder_path, 'unlabeled_data.csv'),
                        help='Path for unlabeled data, Gene names in column header, sample names in the first row')
    parser.add_argument('--labeled_data', type=str,
                        default=os.path.join(data_folder_path, 'labeled_data.csv'),
                        help='Path unlabeled data, Gene names in column header, sample names in the first row')
    parser.add_argument('--data_labels', type=str,
                        default=os.path.join(data_folder_path, 'data_labels.csv'),
                        help='Path for data labels, senitive or resistant with drug names on the column header')
    parser.add_argument('--data_for_hvg', type=str,
                        default=os.path.join(data_folder_path, 'Phase2_StJude_TARGET-10_TALL_protein_coding.csv'),
                        help='Data to determine highly variable genes')
    parser.add_argument('--test_size', type=float,
                        default=0.2,
                        help='The percentage of data to be used for test or validation, default=0.2')
    parser.add_argument('--patience', type=int,
                        default=50,
                        help='Number of epochs with no improvement before early stopping, default=20')
    parser.add_argument('--max_epochs', type=int,
                        default=1000,
                        help='Maximum number of epochs to train for, default=1000')
    parser.add_argument('--random_state', type=int,
                        default=12,
                        help='Random state for reproducibility, default=12')
    parser.add_argument('--n_trials', type=int,
                        default=20,
                        help='Number of trials for hyperparameter search, default=20')
    parser.add_argument('--cv', type=int,
                        default=5,
                        help='Number of cross-validation folds for hyperparameter search, default=5')
    parser.add_argument('--n_top_genes', type=int,
                        default=None,
                        help='the number of highly variable genes, default=None')
    parser.add_argument('--min_disp', type=float,
                        default=1.5,
                        help='the number of min_disp, default=0.5')
    parser.add_argument('--flavor', type=str,
                        default='seurat',
                        help='the number of min_disp, default=seurat, other: seurat_v3, cell_ranger')
    parser.add_argument('--mask_type', type=str,
                        default="sparsemax",
                        help='TabNet parameter "entmax","sparsemax", default="sparsemax"')
    parser.add_argument('--n_d_a', type=int,
                        default=22,
                        help='TabNet parameter, default=22')
    parser.add_argument('--n_steps', type=int,
                        default=4,
                        help='TabNet parameter, default=None')
    parser.add_argument('--gamma', type=float,
                        default=1.2,
                        help='TabNet parameter, default=1.2')
    parser.add_argument('--lambda_sparse', type=float,
                        default=0.00005,
                        help='TabNet parameter, default=s0.00005')
    warnings.filterwarnings("ignore")
    args, _ = parser.parse_known_args()
    return args
