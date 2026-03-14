# Concept Map — ML Topics Across CS229 & Applied ML

> Cross-course mapping of 18 ML concepts showing where each topic appears in CS229 (Stanford) and Applied ML courses.

---

## 1. Linear Regression

| Course | Coverage |
|---|---|
| **CS229** | Ch1 — Linear Regression |
| **Applied ML** | Module 2 |

**Connections:** CS229 Ch1 covers LMS algorithm, normal equations, and probabilistic interpretation. Applied ML Module 2 implements with sklearn.linear_model.LinearRegression.

**Prerequisites:** None

**Keywords:** least squares, normal equations, gradient descent, LinearRegression

---

## 2. Logistic Regression

| Course | Coverage |
|---|---|
| **CS229** | Ch2 — Logistic Regression |
| **Applied ML** | Module 2 |

**Connections:** CS229 Ch2 covers sigmoid function, maximum likelihood, and multi-class softmax. Applied ML Module 2 uses sklearn.linear_model.LogisticRegression.

**Prerequisites:** linear-regression

**Keywords:** sigmoid, cross-entropy, LogisticRegression, classification

---

## 3. K-Nearest Neighbors

| Course | Coverage |
|---|---|
| **CS229** | — |
| **Applied ML** | Module 1 |

**Connections:** Applied ML Module 1 introduces K-NN as first algorithm. Distance-based classification with parameter k.

**Prerequisites:** None

**Keywords:** knn, k-nn, distance, KNeighborsClassifier, lazy learning

---

## 4. Support Vector Machines (SVM)

| Course | Coverage |
|---|---|
| **CS229** | Ch6 — Support Vector Machines |
| **Applied ML** | Module 2 |

**Connections:** CS229 Ch6 covers margins, kernels, duality, and SMO algorithm. Applied ML Module 2 uses sklearn.svm.SVC with kernel trick.

**Prerequisites:** linear-regression, logistic-regression

**Keywords:** support vector machine, margin, kernel, SVC, SVM, rbf

---

## 5. Regularization

| Course | Coverage |
|---|---|
| **CS229** | Ch9 — Regularization |
| **Applied ML** | Module 2 |

**Connections:** CS229 Ch9 theory of L1/L2 regularization, bias-variance tradeoff. Applied ML Module 2 implements Ridge and Lasso regression.

**Prerequisites:** linear-regression

**Keywords:** ridge, lasso, l1, l2, overfitting, regularization

---

## 6. Cross-Validation

| Course | Coverage |
|---|---|
| **CS229** | Ch9 — Regularization |
| **Applied ML** | Module 3 |

**Connections:** CS229 Ch9 covers model selection via cross-validation. Applied ML Module 3 implements with sklearn.model_selection.cross_val_score.

**Prerequisites:** train-test-split

**Keywords:** cross validation, k-fold, model selection, cross_val_score

---

## 7. Decision Trees

| Course | Coverage |
|---|---|
| **CS229** | — |
| **Applied ML** | Module 2 |

**Connections:** Applied ML Module 2 covers decision trees and interpretation. Not in CS229 core content.

**Prerequisites:** supervised-learning

**Keywords:** decision tree, DecisionTreeClassifier, tree, interpretable

---

## 8. Random Forest

| Course | Coverage |
|---|---|
| **CS229** | — |
| **Applied ML** | Module 4 |

**Connections:** Applied ML Module 4 covers ensemble methods. Random Forest is bagging of decision trees.

**Prerequisites:** decision-trees

**Keywords:** random forest, RandomForestClassifier, ensemble, bagging

---

## 9. Gradient Boosting

| Course | Coverage |
|---|---|
| **CS229** | — |
| **Applied ML** | Module 4 |

**Connections:** Applied ML Module 4 covers gradient boosting machines. Sequentially improves weak learners.

**Prerequisites:** decision-trees

**Keywords:** gradient boosting, GradientBoostingClassifier, ensemble, boosting, xgboost

---

## 10. Neural Networks

| Course | Coverage |
|---|---|
| **CS229** | Ch7 — Deep Learning |
| **Applied ML** | Module 4 |

**Connections:** CS229 Ch7 covers MLPs, backpropagation, and deep learning theory. Applied ML Module 4 uses sklearn.neural_network.MLPClassifier.

**Prerequisites:** logistic-regression

**Keywords:** neural network, mlp, deep learning, backpropagation, MLPClassifier

---

## 11. Naive Bayes

| Course | Coverage |
|---|---|
| **CS229** | Ch4 — Generative Algorithms |
| **Applied ML** | Module 2 |

**Connections:** CS229 Ch4 derives Naive Bayes from Bayes' rule with independence assumption. Applied ML Module 2 uses sklearn.naive_bayes.

**Prerequisites:** None

**Keywords:** naive bayes, bayes, GaussianNB, generative

---

## 12. Feature Scaling

| Course | Coverage |
|---|---|
| **CS229** | — |
| **Applied ML** | Module 1, Module 2 |

**Connections:** Applied ML covers StandardScaler and MinMaxScaler. Critical for distance-based and gradient-based algorithms.

**Prerequisites:** None

**Keywords:** scaling, normalization, StandardScaler, MinMaxScaler, preprocessing

---

## 13. Evaluation Metrics

| Course | Coverage |
|---|---|
| **CS229** | — |
| **Applied ML** | Module 3 |

**Connections:** Applied ML Module 3 dedicated to evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

**Prerequisites:** classification

**Keywords:** accuracy, precision, recall, f1, roc, auc, confusion matrix

---

## 14. Bias-Variance Tradeoff

| Course | Coverage |
|---|---|
| **CS229** | Ch8 — Generalization |
| **Applied ML** | Module 2 |

**Connections:** CS229 Ch8 mathematical decomposition of bias-variance tradeoff. Applied ML discusses underfitting vs overfitting.

**Prerequisites:** train-test-split

**Keywords:** bias, variance, overfitting, underfitting, generalization

---

## 15. PCA (Principal Component Analysis)

| Course | Coverage |
|---|---|
| **CS229** | Ch12 — PCA |
| **Applied ML** | Module 4 |

**Connections:** CS229 Ch12 covers eigenvector decomposition and variance maximization. Applied ML Module 4 uses sklearn.decomposition.PCA.

**Prerequisites:** None

**Keywords:** pca, dimensionality reduction, eigenvector, principal component

---

## 16. Clustering

| Course | Coverage |
|---|---|
| **CS229** | Ch10 — Clustering |
| **Applied ML** | Module 4 |

**Connections:** CS229 Ch10 covers K-means algorithm and EM. Applied ML Module 4 uses sklearn.cluster.KMeans and other clustering methods.

**Prerequisites:** None

**Keywords:** clustering, k-means, kmeans, unsupervised, KMeans

---

## 17. Kernels

| Course | Coverage |
|---|---|
| **CS229** | Ch5 — Kernel Methods, Ch6 — Support Vector Machines |
| **Applied ML** | Module 2 |

**Connections:** CS229 Ch5 covers kernel trick and Mercer's theorem. Ch6 applies to SVMs. Applied ML uses RBF kernel in SVC.

**Prerequisites:** svm

**Keywords:** kernel, rbf, kernel trick, feature map

---

## Quick Reference: Coverage Matrix

| Concept | CS229 | Applied ML |
|---|---|---|
| Linear Regression | Ch1 | Module 2 |
| Logistic Regression | Ch2 | Module 2 |
| K-Nearest Neighbors | — | Module 1 |
| SVM | Ch6 | Module 2 |
| Regularization | Ch9 | Module 2 |
| Cross-Validation | Ch9 | Module 3 |
| Decision Trees | — | Module 2 |
| Random Forest | — | Module 4 |
| Gradient Boosting | — | Module 4 |
| Neural Networks | Ch7 | Module 4 |
| Naive Bayes | Ch4 | Module 2 |
| Feature Scaling | — | Module 1-2 |
| Evaluation Metrics | — | Module 3 |
| Bias-Variance | Ch8 | Module 2 |
| PCA | Ch12 | Module 4 |
| Clustering | Ch10 | Module 4 |
| Kernels | Ch5-6 | Module 2 |

> Note: The CONCEPT_MAP in the source contains 17 entries. Some concepts like "train-test-split", "supervised-learning", and "classification" are referenced as prerequisites but defined elsewhere.
