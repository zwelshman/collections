# Data Wrangling and Exploratory Data Analysis

This training module covers the fundamentals of using Python for data wrangling and exploratory data analysis, with a focus on healthcare applications. We will walk through importing data from various sources, cleaning and preprocessing the data, handling missing values, and performing exploratory analysis. The module includes working code examples using Python libraries like Pandas, NumPy, and Matplotlib, applied to a sample dataset of electronic health records.

### 1. Importing Data

Python provides several ways to import data from different sources such as CSV files, databases, and web pages. We'll use the Pandas library to read in a CSV file of patient data.

```python
import pandas as pd

# Read CSV file into a DataFrame
patient_data = pd.read_csv('patient_records.csv')

# Display first 5 rows
print(patient_data.head())
```

Output:

```
   patient_id  age  weight  height  bmi     diagnosis
0           1   45      80     170   27.68  hypertension
1           2   32      65     160   25.39        healthy
2           3   58      90     180   27.78      diabetes
3           4   41      72     165   26.45  hypertension
4           5   29      54     155   22.48        healthy
```

The CSV file `patient_records.csv` contains information on patient demographics and diagnoses. We use `pd.read_csv()` to load the data into a Pandas DataFrame called `patient_data`. The `head()` function displays the first few rows.

### 2. Data Cleaning and Preprocessing

Real-world data often contains inconsistencies, errors, and missing values that need to be addressed before analysis. Let's clean up our patient DataFrame.

```python
# Check for missing values
print(patient_data.isnull().sum())

# Drop rows with missing values
patient_data = patient_data.dropna()

# Convert data types
patient_data['age'] = patient_data['age'].astype(int) 
patient_data['weight'] = patient_data['weight'].astype(float)
patient_data['height'] = patient_data['height'].astype(float)

# Calculate BMI
patient_data['bmi'] = patient_data['weight'] / (patient_data['height']/100)**2
patient_data['bmi'] = patient_data['bmi'].round(2)
```

Output:

```
patient_id    0
age           0
weight        1
height        0
bmi           0
diagnosis     0
dtype: int64
```

We first check for missing values using `isnull().sum()`, which shows there is 1 missing weight value. We remove any rows with `NaN` using `dropna()`.

Next, we ensure the `age`, `weight`, `height` columns are numeric types with `astype()`. Finally, we calculate the body mass index (BMI) using the weight and height values and round it to 2 decimal places.

### 3. Handling Missing Data

In cases where dropping rows with missing data is not feasible, we can fill in the gaps using various techniques. A common approach is to replace missing values with the mean or median of the column.

```python
# Fill missing values with column mean
patient_data['weight'].fillna(patient_data['weight'].mean(), inplace=True)
```

Here we use `fillna()` to replace any missing `weight` values with the average weight across all patients. The `inplace=True` argument modifies the DataFrame directly.

### 4. Exploratory Data Analysis

With our data cleaned, we can start exploring it to uncover insights and trends. Pandas provides functions to quickly summarize the data.

```python
# Get basic statistics
print(patient_data.describe())

# Group by diagnosis and calculate average age
diagnosis_groups = patient_data.groupby('diagnosis').age.mean()
print(diagnosis_groups)
```

Output:

```
       patient_id         age      weight      height         bmi
count  500.000000  500.000000  500.000000  500.000000  500.000000
mean   250.500000   42.130000   71.141400  166.209000   25.740940
std    144.481833   11.356351   12.895377    7.819575    4.024026
min      1.000000   18.000000   45.000000  145.000000   17.360000
25%    125.750000   33.000000   61.000000  160.000000   22.835000
50%    250.500000   42.000000   70.000000  165.000000   25.390000
75%    375.250000   51.000000   80.000000  172.000000   28.405000
max    500.000000   65.000000  110.000000  190.000000   38.570000

diagnosis
diabetes        55.763889
healthy         35.778846
hypertension    49.510204
Name: age, dtype: float64
```

The `describe()` function computes summary statistics for each numeric column like count, mean, min, max, and quartiles.

We can also use `groupby()` to segment the data by categorical variables. Here we group by `diagnosis` and calculate the mean `age` for each condition. This shows patients with diabetes and hypertension tend to be older on average compared to healthy individuals.

### 5. Data Visualization

Visualizing the data is a powerful way to identify patterns and communicate findings. We'll use Matplotlib to create some basic plots.

```python
import matplotlib.pyplot as plt

# Histogram of BMI distribution
plt.figure(figsize=(8,5))
plt.hist(patient_data['bmi'], bins=20)
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('Distribution of Patient BMI')
plt.show()

# Scatter plot of Weight vs Height
plt.figure(figsize=(8,5))
plt.scatter(patient_data['weight'], patient_data['height'])
plt.xlabel('Weight (kg)')  
plt.ylabel('Height (cm)')
plt.title('Patient Weight vs Height')
plt.show()
```

The first plot is a histogram showing the distribution of patient BMI values. We can see that most patients have a BMI between 20-30, with a smaller proportion in the obese range above 30.

The second plot is a scatter plot comparing patient weight and height. There is a general positive correlation between the two variables, with taller patients tending to weigh more. However, there are some outliers that deviate from this trend.

### Conclusion

This training module provided an introduction to using Python for data wrangling and exploratory analysis in a healthcare context. We covered how to import data, perform data cleaning, handle missing values, generate summary statistics, and create visualizations. These techniques form the foundation for deriving valuable insights from patient data to improve care delivery and outcomes.

**Citations:** \[1] https://www.geeksforgeeks.org/data-wrangling-in-python/ \[2] https://www.kaggle.com/code/imoore/intro-to-exploratory-data-analysis-eda-in-python \[3] https://dataheadhunters.com/academy/how-to-implement-patient-data-analysis-in-python-for-healthcare/ \[4] https://www.tutorialspoint.com/python\_data\_science/python\_data\_wrangling.htm
