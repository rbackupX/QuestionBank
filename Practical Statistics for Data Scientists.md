# Topics for "Practical Statistics for Data Scientists". For personal Reference Only.
Copyright © 2019 Safari Books Online.

**Source :** https://www.oreilly.com/library/view/practical-statistics-for/9781491952955/

Practical Statistics for Data Scientists
by Andrew Bruce, Peter Bruce
Statistical methods are a key part of of data science, yet very few data scientists have any formal statistics training. Courses and books on basic statistics rarely cover the topic from a data science perspective. This practical guide explains how to apply various statistical methods to data science, tells you how to avoid their misuse, and gives you advice on what's important and what's not.

Many data science resources incorporate statistical methods but lack a deeper statistical perspective. If you’re familiar with the R programming language, and have some exposure to statistics, this quick reference bridges the gap in an accessible, readable format.

With this book, you’ll learn:

* Why exploratory data analysis is a key preliminary step in data science
* How random sampling can reduce bias and yield a higher quality dataset, even with big data
* How the principles of experimental design yield definitive answers to questions
* How to use regression to estimate outcomes and detect anomalies
* Key classification techniques for predicting which categories a record belongs to
* Statistical machine learning methods that “learn” from data
* Unsupervised learning methods for extracting meaning from unlabeled data


### 1. Exploratory Data Analysis
Elements of Structured Data

* **Rectangular Data**
  * Data Frames and Indexes
  * Non rectangular Data Structures

* **Estimates of Location**
Mean
Median and Robust Estimates
Example: Location Estimates of Population and Murder Rates

* **Estimates of Variability**
Standard Deviation and Related Estimates
Estimates Based on Percentiles
Example: Variability Estimates of State Population

* **Exploring the Data Distribution**
Percentiles and Boxplots
Frequency Table and Histograms
Density Estimates

* **Exploring Binary and Categorical Data**
Mode
Expected Value

* **Correlation**
Scatterplots

* **Exploring Two or More Variables**
Hexagonal Binning and Contours (Plotting Numeric versus Numeric Data)
Two Categorical Variables
Categorical and Numeric Data
Visualizing Multiple Variables


### 2. Data and Sampling Distributions
Random Sampling and Sample Bias
Bias
Random Selection
Size versus Quality: When Does Size Matter?
Sample Mean versus Population Mean

* **Selection Bias**
Regression to the Mean

* **Sampling Distribution of a Statistic**
Central Limit Theorem
Standard Error

* **The Bootstrap**
Resampling versus Bootstrapping

* **Confidence Intervals**

* **Normal Distribution**
Standard Normal and QQ-Plots
Long-Tailed Distributions

* **Student’s t-Distribution**

* **Binomial Distribution**

* **Poisson and Related Distributions**
Poisson Distributions
Exponential Distribution
Estimating the Failure Rate
Weibull Distribution


### 3. Statistical Experiments and Significance Testing
A/B Testing
Why Have a Control Group?
Why Just A/B? Why Not C, D…?
For 
Hypo* **thesis Tests**
The Null Hypothesis
Alternative Hypothesis
One-Way, Two-Way Hypothesis Test

* **Resampling**
Permutation Test
Example: Web Stickiness
Exhaustive and Bootstrap Permutation Test
Permutation Tests: The Bottom Line for Data Science
For 
Stat* **istical Significance and P-Values**
P-Value
Alpha
Type 1 and Type 2 Errors
Data Science and P-Values

* **t-Tests**

* **Multiple Testing**

* **Degrees of Freedom**

* **ANOVA**
F-Statistic
Two-Way ANOVA

* **Chi-Square Test**
Chi-Square Test: A Resampling Approach
Chi-Square Test: Statistical Theory
Fisher’s Exact Test
Relevance for Data Science

* **Multi-Arm Bandit Algorithm**

* **Power and Sample Size**
Sample Size


### 4. Regression and Prediction
Simple Linear Regression
The Regression Equation
Fitted Values and Residuals
Least Squares
Prediction versus Explanation (Profiling)

* **Multiple Linear Regression**
Example: King County Housing Data
Assessing the Model
Cross-Validation
Model Selection and Stepwise Regression
Weighted Regression

* **Prediction Using Regression**
The Dangers of Extrapolation
Confidence and Prediction Intervals
Factor Variables in Regression
Dummy Variables Representation
Factor Variables with Many Levels
Ordered Factor Variables
Interpreting the Regression Equation
Correlated Predictors
Multicollinearity
Confounding Variables
Interactions and Main Effects
Testing the Assumptions: Regression Diagnostics
Outliers
Influential Values
Heteroskedasticity, Non-Normality and Correlated Errors
Partial Residual Plots and Nonlinearity
Polynomial and Spline Regression
Polynomial
Splines
Generalized Additive Models


### 5. Classification
Naive Bayes
Why Exact Bayesian Classification Is Impractical
The Naive Solution
Numeric Predictor Variables

* **Discriminant Analysis**
Covariance Matrix
Fisher’s Linear Discriminant
A Simple Example

* **Logistic Regression**
Logistic Response Function and Logit
Logistic Regression and the GLM
Generalized Linear Models
Predicted Values from Logistic Regression
Interpreting the Coefficients and Odds Ratios
Linear and Logistic Regression: Similarities and Differences
Assessing the Model

* **Evaluating Classification Models**
Confusion Matrix
The Rare Class Problem
Precision, Recall, and Specificity
ROC Curve
AUC
Lift

* **Strategies for Imbalanced Data**
Undersampling
Oversampling and Up/Down Weighting
Data Generation
Cost-Based Classification
Exploring the Predictions


### 6. Statistical Machine Learning
K-Nearest Neighbors
A Small Example: Predicting Loan Default
Distance Metrics
One Hot Encoder
Standardization (Normalization, Z-Scores)
Choosing K
KNN as a Feature Engine
Tree Models
A Simple Example
The Recursive Partitioning Algorithm
Measuring Homogeneity or Impurity
Stopping the Tree from Growing
Predicting a Continuous Value
How Trees Are Used

* **Bagging and the Random Forest**
Bagging
Random Forest
Variable Importance
Hyperparameters
Boosting
The Boosting Algorithm
XGBoost
Regularization: Avoiding Overfitting
Hyperparameters and Cross-Validation

### 7. Unsupervised Learning
Principal Components Analysis
A Simple Example
Computing the Principal Components
Interpreting Principal Components

* **K-Means Clustering**
A Simple Example
K-Means Algorithm
Interpreting the Clusters
Selecting the Number of Clusters
Hierarchical Clustering
A Simple Example
The Dendrogram
The Agglomerative Algorithm
Measures of Dissimilarity
Model-Based Clustering
Multivariate Normal Distribution
Mixtures of Normals
Selecting the Number of Clusters

* **Scaling and Categorical Variables**
Scaling the Variables
Dominant Variables
Categorical Data and Gower’s Distance
Problems with Clustering Mixed Data
