import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import itertools


def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

def get_datasets(std_size=False):
    brca = pd.read_csv('data/Gistic2_CopyNumber_BRCA', sep='\t')
    ovca = pd.read_csv('data/Gistic2_CopyNumber_OVCA', sep='\t')
    brca.set_index('Gene Symbol', inplace=True)
    ovca.set_index('Gene Symbol', inplace=True)
    brca = brca.transpose()
    ovca = ovca.transpose()
    brca['BRCA'] = 1
    ovca['BRCA'] = 0
    print('size brca, ovca', len(brca), len(ovca))
    if not std_size:
        print('normal size', len(brca)+len(ovca))
        df = pd.concat([brca, ovca[1:]])
    elif std_size:
        print('sampled size', len(ovca)+len(ovca))
        df = pd.concat([brca[0:len(ovca)], ovca[1:]])
    print(len(df), len(brca.transpose()-2) + len(ovca.transpose()-2))
    return df

# def plot_coefficients_linSVC(classifier, feature_names, top_features=20):
#     coef = classifier.coef_.ravel()
#     top_positive_coefficients = np.argsort(coef)[-top_features:]
#     top_negative_coefficients = np.argsort(coef)[:top_features]
#     top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
#     # create plot
#     plt.figure(figsize=(15, 5))
#     colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
#     plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
#     feature_names = np.array(feature_names)
#     plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
#     plt.show()




if False:
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    normalized_x_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    normalised_x_test = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns)

    svc_unnorm = SVC(kernel='linear')
    svc_unnorm.fit(X_train, y_train)
    un_score = svc_unnorm.score(X_test, y_test)

    svc_norm = SVC(kernel='linear')
    svc_norm.fit(normalized_x_train, y_train)
    nm_score = svc_norm.score(normalised_x_test, y_test)
    print('unnormalised score =', un_score)
    print('scaled score =', nm_score)
    print('============= PCA')
    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_pca = pca.transform(X)
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)
    print("====================== PCA Analysis")
    for x in [0.34, 0.68, 0.95, 0.997, 0.9999, 0.999999, 0.999999998, 0.999999999997440]:
        pca = PCA(n_components=x)
        pca.fit(X_train)
        X_pca = pca.transform(X)
        print('at', x, '% of the variance ======')
        print("original shape:   ", X.shape)
        print("transformed shape:", X_pca.shape)
        y_red = pd.Series(y)

        df_redd = pd.DataFrame(X_pca)
        svc_reddim = SVC()
        # y_red
        X_traind, X_testd, y_traind, y_testd = train_test_split(df_redd, y)
        svc_reddim.fit(X_traind, y_traind)
        print(svc_reddim.score(X_testd, y_testd))

