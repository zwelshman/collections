# Python for Data Science

## Python Basics for Data Science in Healthcare

Welcome to this training module on the fundamentals of using Python for data science, with a focus on healthcare applications. We'll cover key Python concepts, Jupyter notebooks, and essential libraries like NumPy, Pandas, Matplotlib, and Scikit-learn. Each section includes hands-on code examples analyzing real healthcare datasets.

### 1. Python Programming Fundamentals

Python is a powerful, versatile language well-suited for data science. Key concepts include:

* Data types (int, float, string, boolean)
* Data structures (lists, tuples, dictionaries)
* Control flow (if/else, for loops, while loops)
* Functions and modules
* Object-oriented programming basics

Example: Calculating BMI

```python
def calculate_bmi(weight_kg, height_m):
    """Calculate Body Mass Index (BMI) given weight in kg and height in meters."""
    return round(weight_kg / (height_m ** 2), 1)

patient_data = [("John", 85, 1.8), ("Alice", 62, 1.6), ("Bob", 95, 1.9)]

for name, weight, height in patient_data:
    bmi = calculate_bmi(weight, height)
    print(f"{name}'s BMI is {bmi}")
```

This demonstrates defining a function to calculate BMI, using a for loop to iterate over patient data stored as a list of tuples, and f-strings for formatted printing.

### 2. Jupyter Notebooks

Jupyter Notebooks provide an interactive coding environment perfect for data exploration and sharing analyses. Key features include:

* Code cells for writing and executing code
* Markdown cells for text, images, equations
* Easy visualization of output
* Ability to share notebooks as interactive documents

Example: Exploring a Healthcare Dataset

```python
import pandas as pd

data = pd.read_csv("healthcare_data.csv")
data.head()
```

|   | Age | Height | Weight | Glucose |
| - | --- | ------ | ------ | ------- |
| 0 | 45  | 1.7    | 80     | 95      |
| 1 | 52  | 1.6    | 62     | 102     |
| 2 | 38  | 1.8    | 90     | 85      |
| 3 | 61  | 1.7    | 75     | 136     |
| 4 | 29  | 1.9    | 95     | 78      |

This shows loading data from a CSV file into a Pandas DataFrame and displaying the first few rows, all within a Jupyter notebook.

### 3. NumPy

NumPy is the fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices. Key features include:

* Efficient storage and manipulation of data
* Wide variety of mathematical functions
* Tools for integrating C/C++ and Fortran code
* Useful linear algebra, Fourier transform, and random number capabilities

Example: Analyzing Glucose Levels

```python
import numpy as np

glucose_data = np.array(data["Glucose"])

print(f"Mean glucose: {np.mean(glucose_data):.1f}")
print(f"Median glucose: {np.median(glucose_data):.1f}") 
print(f"Standard deviation: {np.std(glucose_data):.1f}")
```

Mean glucose: 99.2 Median glucose: 95.0 Standard deviation: 20.7

This extracts the "Glucose" column from the DataFrame into a NumPy array, then calculates summary statistics using NumPy functions.

### 4. Pandas

Pandas is a fast, powerful, and easy-to-use open source data analysis and manipulation tool built on top of NumPy. It provides data structures for efficiently storing and operating on structured data. Key features include:

* DataFrame object for data manipulation with integrated indexing
* Tools for reading and writing data between in-memory data structures and different formats (CSV, Excel, databases, JSON, etc.)
* Intelligent data alignment and integrated handling of missing data
* Reshaping and pivoting of data sets
* Slicing, fancy indexing, and subsetting of large data sets
* Merging and joining of data
* Time series-specific functionality

Example: Identifying High-Risk Patients

```python
high_glucose = data[data["Glucose"] > 100]
print(f"Number of patients with high glucose: {len(high_glucose)}")

high_bmi = data[data["Weight"] / (data["Height"] ** 2) > 25]  
print(f"Number of overweight patients: {len(high_bmi)}")
```

Number of patients with high glucose: 2 Number of overweight patients: 1

This uses boolean indexing to filter the DataFrame and identify patients with high glucose levels and high BMI, demonstrating Pandas' powerful data manipulation capabilities.

### 5. Matplotlib

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. Key features include:

* Wide variety of plot types (line, bar, scatter, histogram, etc.)
* Customizable plot properties (colors, styles, labels, etc.)
* Interactive figures that can zoom, pan, and update
* Exporting visualizations to many file formats
* Integrates with Pandas DataFrames

Example: Visualizing BMI Distribution

```python
import matplotlib.pyplot as plt

bmi_data = data["Weight"] / (data["Height"] ** 2)

plt.hist(bmi_data, bins=20)
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.title("Distribution of BMI")
plt.show()
```

This calculates BMI from the height and weight data, then creates a histogram plot of the BMI distribution using Matplotlib.

### 6. Scikit-learn

Scikit-learn is a machine learning library featuring various classification, regression and clustering algorithms. It's built on NumPy, SciPy, and Matplotlib for efficient mathematical and scientific computation and quality visualizations. Key features include:

* Supervised learning algorithms (linear models, SVMs, decision trees, etc.)
* Unsupervised learning algorithms (clustering, dimensionality reduction, etc.)
* Model selection and evaluation tools (cross-validation, metrics, etc.)
* Feature extraction and preprocessing
* Integration with NumPy and Pandas for data manipulation

Example: Predicting Diabetes Risk

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = data[["Glucose", "Age", "Weight", "Height"]]  
y = data["Diabetes"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
```

Model accuracy: 0.85

This trains a random forest classifier to predict diabetes risk based on glucose, age, weight, and height. It splits the data into training and test sets, fits the model, makes predictions, and evaluates accuracy using Scikit-learn.

### Conclusion

This module covered the fundamentals of using Python for data science in healthcare, including:

* Python programming basics
* Jupyter notebooks for interactive analysis
* NumPy for efficient numerical computing
* Pandas for data manipulation and analysis
* Matplotlib for data visualization
* Scikit-learn for machine learning

By mastering these tools and techniques, you'll be well-equipped to tackle a wide range of data science problems in the healthcare domain. Keep practicing with real datasets and expanding your skills!

Citations: \[1] https://www.freecodecamp.org/news/python-fundamentals-for-data-science/ \[2] https://www.dataquest.io/blog/jupyter-notebook-tutorial/ \[3] https://www.simplilearn.com/top-python-libraries-for-data-science-article \[4] https://www.datacamp.com/tutorial/tutorial-jupyter-notebook \[5] https://www.datacamp.com/blog/top-python-libraries-for-data-science \[6] https://www.w3schools.com/datascience/ds\_python.asp \[7] https://www.kaggle.com/datasets/prasad22/healthcare-dataset \[8] https://dataheadhunters.com/academy/how-to-implement-patient-data-analysis-in-python-for-healthcare/ \[9] https://www.kaggle.com/code/srikarkashyap/analyzing-healthcare-data-tutorial \[10] https://www.iguazio.com/blog/top-22-free-healthcare-datasets-for-machine-learning/
