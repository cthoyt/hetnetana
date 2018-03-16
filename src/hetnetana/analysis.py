# -*- coding: utf-8 -*-

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.cross_validation import cross_val_predict
from sklearn.decomposition import PCA  # , FastICA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix

log = logging.getLogger()
log.info('Loaded {}'.format(Axes3D))


# TODO: update to match notebook
def main(footprint_path, training_path, od):
    """

    :param footprint_path: input footprints csv
    :param training_path: input training data csv
    :param od: output directory
    :return:
    """
    plt.rcParams['figure.figsize'] = 10.5, 4

    features = pd.read_csv(footprint_path, index_col=0)
    training_label_df = pd.read_csv(training_path, index_col=0)

    training_data = features.loc[training_label_df.index]
    training_labels = training_label_df['induced']

    """ UNSUPERVISED ROUTE """

    pca = PCA(n_components=None)
    pca.fit(training_data)

    plt.title("PCA Parameter Optimization")
    plt.xlabel("Number Components Used")
    plt.ylabel("Cumulative Explained Variance Ratio")

    plt.xticks(np.arange(pca.n_components_), np.arange(1, pca.n_components_))
    plt.grid(b=True, which='major', color='0.65', linestyle='--')
    plt.axhline(0.8, color='y', linestyle='--', linewidth=2)
    plt.axhline(0.9, color='g', linestyle='--', linewidth=2)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'bo-')

    plt.savefig(os.path.join(od, 'pca_param_optimization.png'))
    plt.clf()

    fig = plt.figure()
    plt.suptitle("Comparison of # PCA Components Kept", fontsize=18)

    ax = plt.subplot(122, projection='3d')
    ax.set_title('PCA Reduction to 3 Components')
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(training_data)
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
               c=['r' if x else 'b' for x in training_labels], s=120, alpha=0.5)

    ax = plt.subplot(121)
    ax.set_title('PCA Reduction to 2 Components')
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(training_data)
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=['r' if x else 'b' for x in training_labels], s=120, alpha=0.5)

    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=1)
    kmeans.fit(reduced_data)
    h = 0.01
    x_min, x_max = reduced_data[:, 0].min() - 0.1, reduced_data[:, 0].max() + 0.1
    y_min, y_max = reduced_data[:, 1].min() - 0.1, reduced_data[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.imshow(Z, interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap=plt.cm.Pastel1,
              aspect='auto', origin='lower')

    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='black', zorder=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.set_tight_layout(dict(pad=5.0))
    plt.savefig(os.path.join(od, 'pca_scatter.png'))
    plt.clf()

    """ SUPERVISED ROUTE """

    kb = SelectKBest(chi2, k='all')
    kb.fit(training_data, training_labels)
    kb_df = pd.DataFrame(kb.scores_, columns=['kb_score'], index=training_data.columns)

    clf = ExtraTreesClassifier()
    clf = clf.fit(training_data, training_labels)
    clf_df = pd.DataFrame(clf.feature_importances_, columns=['clf_score'], index=training_data.columns)

    fdf = pd.concat([clf_df.rank().astype(int), clf_df, kb_df.rank().astype(int), kb_df], axis=1, join='inner')
    fdf.columns = ['clf_rank', 'clf_score', 'kb_rank', 'kb_score']
    fdf = fdf.sort_values('clf_rank', ascending=False)

    fdf['combine_rank'] = (fdf['clf_rank'] + fdf['kb_rank']).astype(int)
    fdf = fdf.sort_values('combine_rank', ascending=False)
    fdf['path_length'] = [1 + x.count('&') for x in fdf.index]
    fdf.to_csv(os.path.join(od, 'feature_scores.csv'), float_format='%.3f')

    # In[6]:
    fig = plt.figure()
    fig.suptitle('Feature Stratification', fontsize=18)

    ax = plt.subplot(121)
    ax.set_title("Feature Score Scatter")
    ax.set_xlabel("K-Best Score")
    ax.set_ylabel("Random Forest Score")
    ax.scatter(fdf['kb_score'], fdf['clf_score'], s=120, alpha=0.6)

    ax = plt.subplot(122)
    ax.set_title("Feature Rank Scatter")
    ax.set_xlabel("K-Best Rank")
    ax.set_ylabel("Random Forest Rank")
    ax.scatter(fdf['kb_rank'], fdf['clf_rank'], s=120, alpha=0.6, c=(fdf['kb_rank'] + fdf['clf_rank']))

    fig.set_tight_layout(dict(pad=5.0))
    fig.savefig(os.path.join(od, 'feature_strat.png'))
    plt.clf()

    # train LR
    selected_features = fdf['combine_rank'].sort_values(ascending=False).head(4).index
    features_cut = training_data[selected_features]

    plt.suptitle("Before and After Feature Selection", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.title('All Features Logistic Regression Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    loo = cross_validation.LeaveOneOut(len(training_labels))
    lr = linear_model.LogisticRegression(C=1e5)
    predicted_labels = cross_val_predict(lr, training_data, training_labels, cv=loo)
    cm = confusion_matrix(training_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['uninteresting', 'interesting'], rotation=45)
    plt.yticks(tick_marks, ['uninteresting', 'interesting'])

    for i, cas in enumerate(cm):
        for j, c in enumerate(cas):
            plt.annotate(c, xy=(j, i), horizontalalignment='center', verticalalignment='center',
                         bbox=dict(fc='w', boxstyle='round,pad=1'))

    plt.subplot(1, 2, 2)
    plt.title('Top Features Logistic Regression Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    loo = cross_validation.LeaveOneOut(len(training_labels))
    lr = linear_model.LogisticRegression(C=1e5)
    predicted_labels = cross_val_predict(lr, features_cut, training_labels, cv=loo)
    cm = confusion_matrix(training_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['uninteresting', 'interesting'], rotation=45)
    plt.yticks(tick_marks, ['uninteresting', 'interesting'])

    for i, cas in enumerate(cm):
        for j, c in enumerate(cas):
            plt.annotate(c, xy=(j, i), horizontalalignment='center', verticalalignment='center',
                         bbox=dict(fc='w', boxstyle='round,pad=1'))

    plt.tight_layout(pad=5.0)
    plt.savefig(os.path.join(od, 'feature_selection_classification_results.png'))
    plt.clf()
