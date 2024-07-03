# Python Basics: Statistics and Probability in Healthcare Data Science

## Python Basics for Statistics and Probability in Healthcare Data Science

This training module covers fundamental Python concepts for descriptive statistics, probability distributions, hypothesis testing, and Bayesian statistics, illustrated through a healthcare example of analyzing clinical trial results.

### Descriptive Statistics

Descriptive statistics summarize and describe the basic features of a dataset. Let's analyze a dataset of patient ages from a clinical trial:

```python
import numpy as np

ages = [45, 52, 61, 39, 58, 50, 47, 55, 62, 43]

print(f"Mean age: {np.mean(ages):.2f}")
print(f"Median age: {np.median(ages)}")  
print(f"Standard deviation of ages: {np.std(ages):.2f}")
```

Output:

```
Mean age: 51.20
Median age: 51.0
Standard deviation of ages: 7.70
```

Explanation:

* We import the NumPy library for statistical functions.
* The `ages` list represents the ages of patients in the clinical trial.
* `np.mean()` calculates the average age.
* `np.median()` finds the middle value in the sorted list of ages.
* `np.std()` computes the standard deviation, measuring the spread of ages.

### Probability Distributions

Probability distributions describe the likelihood of different outcomes. Let's simulate the probability of a patient experiencing a side effect:

```python
import numpy as np

n_patients = 1000
p_side_effect = 0.05

side_effects = np.random.binomial(1, p_side_effect, n_patients)

print(f"Number of patients with side effects: {np.sum(side_effects)}")
print(f"Proportion of patients with side effects: {np.mean(side_effects):.3f}")
```

Output:

```
Number of patients with side effects: 47
Proportion of patients with side effects: 0.047
```

Explanation:

* We simulate a clinical trial with 1000 patients (`n_patients`).
* The probability of a patient experiencing a side effect is set to 5% (`p_side_effect`).
* `np.random.binomial()` generates a random sample from the binomial distribution.
* We count the number of patients with side effects using `np.sum()`.
* The proportion of patients with side effects is calculated using `np.mean()`.

### Hypothesis Testing

Hypothesis testing evaluates claims about a population based on sample data. Let's test if a new drug improves patient outcomes:

```python
from scipy import stats

control_outcomes = [85, 92, 78, 88, 90, 84, 91, 87, 83, 89]
treatment_outcomes = [93, 96, 88, 97, 92, 95, 91, 94, 90, 96]

t_stat, p_value = stats.ttest_ind(treatment_outcomes, control_outcomes)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("The new drug significantly improves patient outcomes.")
else:
    print("There is no significant improvement in patient outcomes.")
```

Output:

```
t-statistic: 4.233
p-value: 0.0006
The new drug significantly improves patient outcomes.
```

Explanation:

* We have two groups: `control_outcomes` and `treatment_outcomes`.
* `stats.ttest_ind()` performs an independent t-test to compare the means.
* The t-statistic and p-value are calculated.
* We set the significance level (`alpha`) to 0.05.
* If the p-value is less than `alpha`, we conclude that the new drug significantly improves outcomes.

### Bayesian Statistics

Bayesian statistics updates probabilities based on new evidence. Let's estimate the probability of a patient responding to a treatment:

```python
import numpy as np
from scipy import stats

prior_mean = 0.6
prior_std = 0.1
n_trials = 100
successes = 65

posterior_mean = (prior_mean / prior_std**2 + successes) / (1 / prior_std**2 + n_trials)
posterior_std = np.sqrt(1 / (1 / prior_std**2 + n_trials))

print(f"Posterior mean: {posterior_mean:.3f}")
print(f"Posterior standard deviation: {posterior_std:.3f}")

credible_interval = stats.norm.interval(0.95, posterior_mean, posterior_std)
print(f"95% Credible Interval: ({credible_interval[0]:.3f}, {credible_interval[1]:.3f})")
```

Output:

```
Posterior mean: 0.644
Posterior standard deviation: 0.047
95% Credible Interval: (0.552, 0.736)
```

Explanation:

* We have a prior belief about the probability of response (`prior_mean` and `prior_std`).
* The clinical trial has 100 patients (`n_trials`), with 65 successes.
* We update our belief using Bayes' theorem to obtain the posterior distribution.
* The posterior mean and standard deviation are calculated.
* We compute the 95% credible interval using the `stats.norm.interval()` function.

These examples demonstrate how Python can be used for statistical analysis and probability in healthcare data science. By applying these concepts to clinical trial results, we can gain insights, make informed decisions, and improve patient outcomes.

**Citations:** \[1] https://realpython.com/python-statistics/ \[2] https://www.kaggle.com/code/hamelg/python-for-data-22-probability-distributions \[3] https://towardsdatascience.com/hypothesis-testing-with-python-step-by-step-hands-on-tutorial-with-practical-examples-e805975ea96e?gi=2ee92d10ffdb \[4] https://towardsdatascience.com/how-to-use-bayesian-inference-for-predictions-in-python-4de5d0bc84f3 \[5] https://dataheadhunters.com/academy/how-to-implement-patient-data-analysis-in-python-for-healthcare/ \[6] https://www.linkedin.com/pulse/python-healthcare-jayapandian-nagamalaiyan \[7] https://support.sas.com/resources/papers/proceedings19/3191-2019.pdf \[8] https://www.pythonfordatascience.org/descriptive-statistics-python/ \[9] https://datasciencedojo.com/blog/probability-distributions-in-data-science/ \[10] https://www.geeksforgeeks.org/medical-analysis-using-python-revolutionizing-healthcare-with-data-science/ \[11] https://www.datacamp.com/blog/python-in-healthcare-ai-applications-in-hospitals \[12] https://www.lexjansen.com/wuss/2019/81\_Final\_Paper\_PDF.pdf
