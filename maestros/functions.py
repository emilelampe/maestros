import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_consistent_length, column_or_1d
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from itertools import combinations
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt


def multilabel_stratified_group_split(
    X, y, groups, test_size=0.2, random_state=None, shuffle=True
):
    """
    Splits a dataset into train and test sets while preserving the multilabel distribution for each group.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The feature matrix.
    y : array-like, shape (n_samples, n_labels)
        The multilabel target matrix.
    groups : array-like, shape (n_samples,)
        The group labels for each sample.
    test_size : float, optional (default=0.2)
        The proportion of the dataset to include in the test split.
    random_state : int or RandomState instance, optional (default=None)
        The random number generator to use for shuffling.
    shuffle : bool, optional (default=True)
        Whether to shuffle the input samples before splitting.
    
    Returns
    -------
    X_train : array-like, shape (n_train_samples, n_features)
        The training feature matrix.
    X_test : array-like, shape (n_test_samples, n_features)
        The test feature matrix.
    y_train : array-like, shape (n_train_samples, n_labels)
        The training multilabel target matrix.
    y_test : array-like, shape (n_test_samples, n_labels)
        The test multilabel target matrix.
    train_indices : array-like, shape (n_train_samples,)
        The indices of the training samples.
    test_indices : array-like, shape (n_test_samples,)
        The indices of the test samples.
    """

    # Check input
    X = check_array(X, "csr", ensure_min_features=2)
    y = check_array(y, "csr", ensure_2d=False, ensure_min_features=2)
    groups = column_or_1d(groups, warn=True)
    check_consistent_length(X, y, groups)

    target_type = type_of_target(y)
    if target_type != "multilabel-indicator":
        raise ValueError(
            f"Multilabel stratification is not supported for target type {target_type}"
        )

    # Check that test_size is within the valid range
    if not (0 < test_size < 1):
        raise ValueError(f"Test size must be between 0 and 1, got {test_size}")

    rng = check_random_state(random_state)
    group_indices = np.arange(groups.shape[0])
    
    if shuffle:
        rng.shuffle(group_indices)
    
    unique_groups, group_counts = np.unique(groups[group_indices], return_counts=True)

    mskf = MultilabelStratifiedKFold(n_splits=int(1/test_size), random_state=random_state, shuffle=shuffle)
    
    # Create a 2D array representing the labels for each unique group
    group_labels = np.zeros((unique_groups.shape[0], y.shape[1]))
    for idx, group in enumerate(unique_groups):
        group_labels[idx] = np.any(y[groups == group], axis=0)
    
    # Assign train and test indices based on shuffled groups
    for train_group_idx, test_group_idx in mskf.split(unique_groups, group_labels):
        train_indices = group_indices[np.isin(groups[group_indices], unique_groups[train_group_idx])]
        test_indices = group_indices[np.isin(groups[group_indices], unique_groups[test_group_idx])]
        break

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices], train_indices, test_indices

def label_distribution(y):
    """
    Calculate the distribution of labels in a multilabel target matrix.
    
    Parameters
    ----------
    y : array-like, shape (n_samples, n_labels)
        The multilabel target matrix.
        
    Returns
    -------
    distribution : array-like, shape (n_labels,)
        The distribution of labels.
    """
    return np.sum(y, axis=0) / y.shape[0]

def stratification_report(y, y_train, y_test, y_val=None, labels=None):
    """
    Print a stratification report comparing the label distributions in the complete dataset, training set, and test set.
    
    Parameters
    ----------
    y : array-like, shape (n_samples, n_labels)
        The complete multilabel target matrix.
    y_train : array-like, shape (n_train_samples, n_labels)
        The training multilabel target matrix.
    y_test : array-like, shape (n_test_samples, n_labels)
        The test multilabel target matrix.
    y_val : array-like, shape (n_val_samples, n_labels), optional (default=None)
        The validation multilabel target matrix.
    labels : list of str, optional (default=None)
        The names of the labels. If None, generic names will be used.
    """
    complete_set_distribution = label_distribution(y)
    train_set_distribution = label_distribution(y_train)
    test_set_distribution = label_distribution(y_test)
    if y_val is not None:
        val_set_distribution = label_distribution(y_val)

    print("\nLabel distribution:")
    header = "{:<16} {:<10} {:<10}".format("Label", "Complete", "Train")
    if y_val is not None:
        header += " {:<10}".format("Val")
    header += " {:<10}".format("Test")
    print(header)

    for i, (complete, train, test) in enumerate(zip(complete_set_distribution, train_set_distribution, test_set_distribution)):
        if y_val is not None:
            val = val_set_distribution[i]
        else:
            val = ""

        if labels:
            row = f"{labels[i]:<15} {complete:.3f}{' '*5} {train:.3f}{' '*5}"
        else:
            row = f"Label {i:<10} {complete:.3f}{' '*5} {train:.3f}{' '*5}"
        if y_val is not None:
            row += f"{val:.3f}{' '*5}"
        row += f"{test:.3f}"
        print(row)

    print("\nDifferences:")
    header_diff = "{:<16} {:<15}".format("Label", "Train-Complete")
    if y_val is not None:
        header_diff += " {:<15}".format("Val-Complete")
    header_diff += " {:<15}".format("Test-Complete")
    print(header_diff)

    for i, (train_diff, test_diff) in enumerate(zip(np.abs(train_set_distribution - complete_set_distribution), np.abs(test_set_distribution - complete_set_distribution))):
        if y_val is not None:
            val_diff = np.abs(val_set_distribution[i] - complete_set_distribution[i])
        else:
            val_diff = ""

        if labels:
            row_diff = f"{labels[i]:<20} {train_diff:.3f}{' '*10}"
        else:
            row_diff = f"Label {i:<10} {train_diff:.3f}{' '*10}"
        if y_val is not None:
            row_diff += f"{val_diff:.3f}{' '*10}"
        row_diff += f"{test_diff:.3f}"
        print(row_diff)

    print("\nMean Differences:")
    mean_train_diff = np.mean(np.abs(train_set_distribution - complete_set_distribution))
    if y_val is not None:
        mean_val_diff = np.mean(np.abs(val_set_distribution - complete_set_distribution))
    mean_test_diff = np.mean(np.abs(test_set_distribution - complete_set_distribution))

    print(f"Train-Complete: {mean_train_diff:.3f}")
    if y_val is not None:
        mean_val_diff = np.mean(np.abs(val_set_distribution - complete_set_distribution))
        print(f"Val-Complete: {mean_val_diff:.3f}")
    print(f"Test-Complete: {mean_test_diff:.3f}")

def check_disjoint_groups(train_indices, test_indices, groups):
    """
    Check if the groups in the train set and test set are disjoint.
    
    Parameters
    ----------
    train_indices : array-like, shape (n_train_samples,)
        The indices of the training set.
    test_indices : array-like, shape (n_test_samples,)
        The indices of the test set.
    groups : array-like, shape (n_samples,)
        The groups for each sample.

    Raises
    ------
    AssertionError
        If the train and test groups are not disjoint.
    """
    train_groups = groups[train_indices]
    test_groups = groups[test_indices]
    assert np.intersect1d(train_groups, test_groups).size == 0

def create_stratification_chart(y, y_train, y_test, y_val=None, labels=None):
    """
    Create a bar chart comparing the label distributions in the complete dataset, training set, and test set.
    
    Parameters
    ----------
    y : array-like, shape (n_samples, n_labels)
        The complete multilabel target matrix.
    y_train : array-like, shape (n_train_samples, n_labels)
        The training multilabel target matrix.
    y_test : array-like, shape (n_test_samples, n_labels)
        The test multilabel target matrix.
    y_val : array-like, shape (n_val_samples, n_labels), optional (default=None)
        The validation multilabel target matrix.
    labels : list of str, optional (default=None)
        The names of the labels. If None, generic names will be used.
    """

    def prep_df(df, name):
        df = df.stack().reset_index()
        df.columns = ['c1', 'c2', 'values']
        df['Class'] = name
        return df

    fig_w = 6
    fig_h = 4

    n_labels = y.shape[1]

    if y_val is not None:
        sets = {
        'Complete': y,
        'Train': y_train,
        'Validation': y_val,
        'Test': y_test
    }
    else:
        sets = {
        'Complete': y,
        'Train': y_train,
        'Test': y_test
    }

    dict0 = {}
    dict1 = {}

    for s in sets:
        dict0[s] = []
        dict1[s] = []
        for l in range(n_labels):
            dict0[s].append((sets[s][:,l] == 0).astype(int).sum())
            dict1[s].append((sets[s][:,l] == 1).astype(int).sum())

    if labels is None:
        labels = [('L'+str(i)) for i in range(n_labels)]

    df0 = pd.DataFrame(dict0, index=labels, columns=dict0.keys())
    df1 = pd.DataFrame(dict1, index=labels, columns=dict1.keys())

    df0 = prep_df(df0, '0')
    df1 = prep_df(df1, '1')

    df = pd.concat([df0, df1])

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bar_width = 0.6
    opacity = 1

    set_names = list(sets.keys())
    n_sets = len(set_names)

    for i, label in enumerate(labels):
        for j, set_name in enumerate(set_names):
            class_0 = df.loc[(df['c1'] == label) & (df['Class'] == '0') & (df['c2'] == set_name), 'values'].values[0]
            class_1 = df.loc[(df['c1'] == label) & (df['Class'] == '1') & (df['c2'] == set_name), 'values'].values[0]
            total = class_0 + class_1
            class_0_pct = class_0 / total
            class_1_pct = class_1 / total

            xpos = i * (n_sets + 1) + j
            ax.bar(xpos, class_1_pct, bar_width, alpha=opacity, color='#fdbb84', label='1' if i == 0 and j == 0 else "")
            ax.bar(xpos, class_0_pct, bar_width, alpha=opacity, color='#e34a33', bottom=class_1_pct, label='0' if i == 0 and j == 0 else "")

    ax.set_ylim(0, 1)
    
    ax.set_xticks([i * (n_sets + 1) + ((n_sets - 1) / 2) for i in range(len(labels))])
    ax.set_xticklabels([''] * len(labels))
    
    for i, label in enumerate(labels):
        xpos = i * (n_sets + 1) + ((n_sets - 1) / 2)
        ypos = 1.01
        ax.text(xpos, ypos, label, ha='center', va='bottom', fontsize=10, transform=ax.get_xaxis_transform())
        
        for j, set_name in enumerate(set_names):
            xpos = i * (n_sets + 1) + j
            ypos = - fig_w * 0.005
            ax.text(xpos, ypos, set_name, ha='center', va='top', rotation=90, fontsize=(fig_w*1.2), transform=ax.get_xaxis_transform())

    # Make the top bar and right border invisible
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Edit axis colors
    ax.spines['bottom'].set_color('#808080')
    ax.spines['left'].set_color('#808080')
    ax.xaxis.label.set_color('#808080')
    ax.yaxis.label.set_color('#808080')
    ax.tick_params(axis='x', colors='#808080', which='both')
    ax.tick_params(axis='y', which='both', labelcolor='black')

    ax.yaxis.set_tick_params(color='#808080')
    ax.tick_params(axis='x', direction='out', length=5, width=0.5, color='#808080', pad=0, bottom=True, top=False)
    ax.set_xticks([i * (n_sets + 1) + j for i in range(len(labels)) for j in range(n_sets)])

    ax.set_yticks([i/10 for i in range(11)])

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.subplots_adjust(left=0.08, right=0.85, top=0.94, bottom=0.18)

    plt.show()

def create_mock_data(n_samples=100, n_labels=5, n_features=4, n_groups=5, seed=42):
    """
    Create mock data for testing the multilabel stratified group split functions.
    
    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples in the dataset.
    n_labels : int, optional (default=5)
        The number of labels in the multilabel target matrix.
    n_features : int, optional (default=4)
        The number of features in the feature matrix.
    n_groups : int, optional (default=5)
        The number of groups in the dataset.
    seed : int, optional (default=42)
        The seed for the random number generator.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        The feature matrix.
    y : array-like, shape (n_samples, n_labels)
        The multilabel target matrix.
    groups: array-like, shape (n_samples,)
        The groups for each sample.
    """
    # Create a random seed for reproducibility
    np.random.seed(seed)

    # Generate X (feature matrix)
    X = np.random.randn(n_samples, n_features)

    # Generate y (multilabel target matrix)
    all_labels = list(combinations(range(n_labels), n_labels - 1))
    chosen_labels = np.random.choice(len(all_labels), n_samples)
    labels = [all_labels[i] for i in chosen_labels]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    # Generate groups
    groups = np.random.randint(0, n_groups, n_samples)

    # Return mock data
    return X, y, groups