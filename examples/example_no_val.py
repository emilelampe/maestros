from maestros.functions import *

# Create multi-label samples, each belonging to one of the 5 groups
X, y, groups = create_mock_data(n_samples=500, n_labels=5, n_features=4, n_groups=5, seed=42)

# Define the labels that should be displayed in the report and the chart
labels = ['Beach', 'Water', 'Tree', 'House', 'Mountain']

# Perform the multilabel stratified group split
X_train, X_test, y_train, y_test, train_indices, test_indices = multilabel_stratified_group_split(X, y, groups, test_size=0.2, random_state=42, shuffle=True)

# Check if groups in train and test sets are disjoint
check_disjoint_groups(train_indices, test_indices, groups)

# Print stratification report
print(stratification_report(y, y_train, y_test, labels=labels))


# Plot the stratification chart
create_stratification_chart(y, y_train, y_test, labels=labels)