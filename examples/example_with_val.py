from maestros.functions import *

# Create multi-label samples, each belonging to one of the 5 groups
X, y, groups = create_mock_data(n_samples=500, n_labels=5, n_features=4, n_groups=5, seed=42)

# Define the labels that should be displayed in the report and the chart
labels = ['Beach', 'Water', 'Tree', 'House', 'Mountain']

# First, split the data into 80% train+val and 20% test
X_train_val, X_test, y_train_val, y_test, train_val_indices, test_indices = multilabel_stratified_group_split(X, y, groups, test_size=0.2, random_state=42, shuffle=True)

groups_train_val = groups[train_val_indices]

# Next, split the train+val data into 75% train and 25% validation (resulting in 60% train, 20% val, and 20% test)
X_train, X_val, y_train, y_val, train_indices, val_indices = multilabel_stratified_group_split(X_train_val, y_train_val, groups[train_val_indices], test_size=0.25, random_state=42, shuffle=True)

# Print stratification report with the validation set
stratification_report(y, y_train, y_test, y_val=y_val, labels=labels)

# Plot the stratification chart with the validation set
create_stratification_chart(y, y_train, y_test, y_val=y_val, labels=labels)