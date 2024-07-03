# Python Basics: Machine Learning with Scikit-learn

## Python Basics for Machine Learning with Scikit-learn

This training module provides an introduction to using Python and the Scikit-learn library for machine learning, with a focus on healthcare applications. We'll cover the key concepts and algorithms for both supervised and unsupervised learning, along with practical code examples.

### Supervised Learning Algorithms

Supervised learning involves training a model on labeled data to make predictions. The key supervised learning algorithms we'll cover are:

* Linear Regression
* Logistic Regression
* Decision Trees
* Support Vector Machines

#### Linear Regression

Linear regression models the relationship between input features and a continuous output variable. It's commonly used for predicting things like medical costs. Here's an example using the diabetes dataset:

```python
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20] 
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model 
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
```

```
Coefficients: 
[938.23786125]
Mean squared error: 2548.07
Coefficient of determination: 0.47
```

This example uses a single feature from the diabetes dataset to predict disease progression. It splits the data into train and test sets, fits a linear regression model, and evaluates the predictions using mean squared error and R-squared.

#### Logistic Regression

Logistic regression is used for binary classification problems, where the output is one of two classes. An example healthcare application is predicting hospital readmissions\[1]:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assume X contains features and y is binary readmission labels 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
```

This loads a dataset of patient features and readmission labels, trains a logistic regression model, and evaluates the accuracy of its predictions on held-out test data. Logistic regression provides predicted probabilities of each class.

#### Decision Trees

Decision trees learn a series of if-then rules to make predictions. They work for both regression and classification. An example application is detecting insurance fraud\[4]:

```python
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assume X contains claim features and y is binary fraud labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) 

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))
```

This trains a decision tree on insurance claims data to predict if a claim is fraudulent or not. Decision trees are easy to interpret and visualize.

### Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. Two key algorithms are:

* K-Means Clustering
* Principal Component Analysis (PCA)

#### K-Means Clustering

K-means groups data points into K clusters based on similarity. It's often used for customer segmentation. Here's an example of segmenting diabetes patients\[3]:

```python
from sklearn.cluster import KMeans

# Assume X contains patient features 
kmeans = KMeans(n_clusters=4, random_state=0) 
kmeans.fit(X)

print("Cluster centers:\n", kmeans.cluster_centers_)  
print("Labels:\n", kmeans.labels_)
```

This groups the patients into 4 clusters based on their features. The cluster centers represent the average feature values for each group. The labels show which cluster each patient belongs to. This could be used to tailor treatments to different patient segments.

#### Principal Component Analysis

PCA is a dimensionality reduction technique that summarizes a dataset using a smaller number of features. It's useful for compressing data and visualization.

```python
from sklearn.decomposition import PCA

# Assume X contains high-dimensional patient data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X) 

print("Explained variance ratio: ", pca.explained_variance_ratio_)  
print("Principal components:\n", pca.components_)
```

This reduces the patient dataset to 2 principal components. The explained variance shows how much information is retained. The components represent the main directions of variation. Plotting the transformed data can reveal interesting structure.

### Model Evaluation and Selection

Evaluating model performance is critical for selecting and optimizing models. Key techniques include:

* Train/test splitting
* Cross-validation
* Hyperparameter tuning
* Metrics like accuracy, precision, recall, F1-score, ROC curves

Proper evaluation helps ensure models generalize well to new data. Scikit-learn provides many tools to streamline the model building process.

This overview covered the key concepts and algorithms for machine learning in Python with Scikit-learn. The healthcare examples showed how these techniques can be applied to real-world problems. With this foundation, you're ready to start building your own machine learning models!

Citations: \[1] https://scikit-learn.org/stable/tutorial/basic/tutorial.html \[2] https://www.datacamp.com/tutorial/machine-learning-python \[3] https://towardsdatascience.com/predicting-hospital-readmission-for-patients-with-diabetes-using-scikit-learn-a2e359b15f0?gi=eff2b92da90d \[4] https://www.wipro.com/analytics/comparative-analysis-of-machine-learning-techniques-for-detectin/ \[5] https://www.freecodecamp.org/news/customer-segmentation-python-machine-learning/ \[6] https://scikit-learn.org/stable/auto\_examples/linear\_model/plot\_ols.html \[7] https://www.datacamp.com/tutorial/understanding-logistic-regression-python \[8] https://scikit-learn.org/stable/auto\_examples/index.html \[9] https://scikit-learn.org/stable/supervised\_learning.html
