# Top data science questions and answers for 2018:
________________________________________
### General Questions

* Suppose you’re given millions of users that each have hundreds of transactions and these millions of transactions are for tens of thousands of products. How would you group the users together in meaningful segments?

* How would you approach a categorical feature with high-cardinality?

**Answer** :
 An elegant way to include such high cardinality attributes is by transforming the nominal variable to a single continuous one, whose values are correlated with the target label.
  https://www.kdnuggets.com/2016/08/include-high-cardinality-attributes-predictive-model.html

* Describe a project you’ve worked on and how it made a difference.

* What would you do to summarize a Twitter feed?
* What are the steps for wrangling and cleaning data before applying machine learning algorithms?

* How do you measure distance between data points?

  * Minkowski distance
  * Euclidean distance.  
  * Manhattan distance.
  * Mahalanobis distance**

  Distance between categorical data points
  * Hamming distance:
      noOfMatchAttributes / noOfAttributes
  * Jaccard similarity:
      noOfOnesInBoth / (noOfOnesInA + noOfOnesInB - noOfOnesInAandB)

  Distance between mixed categorical and numeric data points
  * distance final = α.distance numeric + (1- α).distance categorical

  Distance between sequence (String, TimeSeries)
    * **edit distance** Basically, *edit distance* reveals how many "modifications" (which can be insert, modify, delete) are needed to change stringA into stringB. This is usually calculated by using thedynamic programming technique.

  Distance between nodes in a network
  Distance between population distribution

https://dzone.com/articles/machine-learning-measuring

* Define variance.

Describe the differences between and use cases for box plots and histograms.

What features would you use to build a recommendation algorithm for users?

Pick any product or app that you really like and describe how you would improve it.
* How would you find an anomaly in a distribution?
Individual Detection Algorithms:

Linear Models for Outlier Detection:

PCA: Principal Component Analysis use the sum of weighted projected distances to the eigenvector hyperplane as the outlier outlier scores) [10]
MCD: Minimum Covariance Determinant (use the mahalanobis distances as the outlier scores) [11, 12]
One-Class Support Vector Machines [3]
Proximity-Based Outlier Detection Models:

LOF: Local Outlier Factor [1]
CBLOF: Clustering-Based Local Outlier Factor [15]
HBOS: Histogram-based Outlier Score [5]
kNN: k Nearest Neighbors (use the distance to the kth nearest neighbor as the outlier score) [13]
Average kNN or kNN Sum Outlier Detection (use the average distance to k nearest neighbors as the outlier score or sum all k distances) [14]
Median kNN Outlier Detection (use the median distance to k nearest neighbors as the outlier score)
Probabilistic Models for Outlier Detection:

ABOD: Angle-Based Outlier Detection [7]
FastABOD: Fast Angle-Based Outlier Detection using approximation [7]
Outlier Ensembles and Combination Frameworks

Isolation Forest [2]
Feature Bagging [9]
Neural Networks and Deep Learning Models (implemented in Keras)

AutoEncoder with Fully Connected NN [16, Chapter 3]
Reference
[1] Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J., 2000, May. LOF: identifying density-based local outliers. In ACM SIGMOD Record, pp. 93-104. ACM.

[2] Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008, December. Isolation forest. In ICDM '08, pp. 413-422. IEEE.

[3] Ma, J. and Perkins, S., 2003, July. Time-series novelty detection using one-class support vector machines. In IJCNN' 03, pp. 1741-1745. IEEE.

[4] Y. Zhao and M.K. Hryniewicki, "DCSO: Dynamic Combination of Detector Scores for Outlier Ensembles," ACM SIGKDD Workshop on Outlier Detection De-constructed (ODD v5.0), 2018.

[5] Goldstein, M. and Dengel, A., 2012. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm. In KI-2012: Poster and Demo Track, pp.59-63.

[6] Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms for outlier ensembles.ACM SIGKDD Explorations Newsletter, 17(1), pp.24-47.

[7] Kriegel, H.P. and Zimek, A., 2008, August. Angle-based outlier detection in high-dimensional data. In KDD '08, pp. 444-452. ACM.

[8] Y. Zhao and M.K. Hryniewicki, "XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning," IEEE International Joint Conference on Neural Networks, 2018.

[9] Lazarevic, A. and Kumar, V., 2005, August. Feature bagging for outlier detection. In KDD '05. 2005.

[10] Shyu, M.L., Chen, S.C., Sarinnapakorn, K. and Chang, L., 2003. A novel anomaly detection scheme based on principal component classifier. MIAMI UNIV CORAL GABLES FL DEPT OF ELECTRICAL AND COMPUTER ENGINEERING.

[11] Rousseeuw, P.J. and Driessen, K.V., 1999. A fast algorithm for the minimum covariance determinant estimator. Technometrics, 41(3), pp.212-223.

[12] Hardin, J. and Rocke, D.M., 2004. Outlier detection in the multiple cluster setting using the minimum covariance determinant estimator. Computational Statistics & Data Analysis, 44(4), pp.625-638.

[13] Ramaswamy, S., Rastogi, R. and Shim, K., 2000, May. Efficient algorithms for mining outliers from large data sets. ACM Sigmod Record, 29(2), pp. 427-438).

[14] Angiulli, F. and Pizzuti, C., 2002, August. Fast outlier detection in high dimensional spaces. In European Conference on Principles of Data Mining and Knowledge Discovery pp. 15-27.

[15] He, Z., Xu, X. and Deng, S., 2003. Discovering cluster-based local outliers. Pattern Recognition Letters, 24(9-10), pp.1641-1650.

[16] Aggarwal, C.C., 2015. Outlier analysis. In Data mining (pp. 237-263). Springer, Cham.

https://github.com/yzhao062/Pyod#quick-start-for-outlier-detection
https://www.datascience.com/blog/python-anomaly-detection
http://aqibsaeed.github.io/2016-07-17-anomaly-detection/

How would you go about investigating if a certain trend in a distribution is due to an anomaly?
How would you estimate the impact Uber has on traffic and driving conditions?
What metrics would you consider using to track if Uber’s paid advertising strategy to acquire new customers actually works? How would you then approach figuring out an ideal customer acquisition cost?

(Data Engineer) Can you explain what REST is?
Machine Learning Questions

Why do you use feature selection?
What is the effect on the coefficients of logistic regression if two predictors are highly correlated? What are the confidence intervals of the coefficients?
What’s the difference between Gaussian Mixture Model and K-Means?
How do you pick k for K-Means?
How do you know when Gaussian Mixture Model is applicable?
Assuming a clustering model’s labels are known, how do you evaluate the performance of the model?

What’s an example of a machine learning project you’re proud of?
Choose any machine learning algorithm and describe it.
Describe how Gradient Boosting works.
Describe the decision tree model.
What is a neural network?
Explain the Bias-Variance Tradeoff
How do you deal with unbalanced binary classification?
What’s the difference between L1 and L2 regularization?

What sort features could you give an Uber driver to predict if they will accept a ride request or not? What supervised learning algorithm would you use to solve the problem and how would compare the results of the algorithm?

Name and describe three different kernel functions and in what situation you would use each.
Describe a method used in machine learning.
How do you deal with sparse data?

How do you prevent overfitting?
How do you deal with outliers in your data?
How do you analyze the performance of the predictions generated by regression models versus classification models?
How do you assess logistic regression versus simple linear regression models?
What’s the difference between supervised learning and unsupervised learning?
What is cross-validation and why would you use it?
What’s the name of the matrix used to evaluate predictive models?
What relationships exist between a logistic regression’s coefficient and the Odds Ratio?
What’s the relationship between Principal Component Analysis (PCA) and Linear & Quadratic Discriminant Analysis (LDA & QDA)
If you had a categorical dependent variable and a mixture of categorical and continuous independent variables, what algorithms, methods, or tools would you use for analysis?
(Business Analytics) What’s the difference between logistic and linear regression? How do you avoid local minima?

What data and models would would you use to measure attrition/churn? How would you measure the performance of your models?
Explain a machine learning algorithm as if you’re talking to a non-technical person.

How would you build a model to predict credit card fraud?
How do you handle missing or bad data?
How would you derive new features from features that already exist?
If you’re attempting to predict a customer’s gender, and you only have 100 data points, what problems could arise?
Suppose you were given two years of transaction history. What features would you use to predict credit risk?
Design an AI program for Tic-tac-toe

Explain overfitting and what steps you can take to prevent it.
Why does SVM need to maximize the margin between support vectors?
Statistics and Probability Questions

Explain Cross-validation as if you’re talking to a non-technical person.
Describe a non-normal probability distribution and how to apply it.

Explain what heteroskedasticity is and how to solve it

Given Twitter user data, how would you measure engagement?

What are some different Time Series forecasting techniques?
Explain Principle Component Analysis (PCA) and equations PCA uses.
How do you solve Multicollinearity?
Write an equation that would optimize the ad spend between Twitter and Facebook.

What’s the probability you’ll draw two cards of the same suite from a single deck?

What are p-values and confidence intervals?

(Data Analyst) If you have 70 red marbles, and the ratio of green to red marbles is 2 to 7, how many green marbles are there?
What would the distribution of daily commutes in New York City look like?
Given a die, would it be more likely to get a single 6 in six rolls, at least two 6s in twelve rolls, or at least one-hundred 6s in six-hundred rolls?

What’s the Central Limit Theorem, and how do you prove it? What are its applications?
Programming and Algorithms

.
(Data Analyst) Write a program that can determine the height of an arbitrary binary tree

Create a function that checks if a word is a palindrome.

Build a power set.
How do you find the median of a very large dataset?

.
Code a function that calculates the square root (2-point precision) of a given number. Follow up: Avoid redundant calculations by now optimizing your function with a caching mechanism.

Suppose you’re given two binary strings, write a function adds them together without using any builtin string-to-int conversion or parsing tools. For example, if you give your function binary strings 100 and 111, it should return 1011. What’s the space and time complexity of your solution?
Write a function that accepts two already sorted lists and returns their union in a sorted list.

Write some code that will determine if brackets in a string are balanced
How do you find the second largest element in a Binary Search Tree?
Write a function that takes two sorted vectors and returns a single sorted vector.
If you have an incoming stream of numbers, how would you find the most frequent numbers on-the-fly?
Write a function that raises one number to another number, i.e. the pow() function.
Split a large string into valid words and store them in a dictionary. If the string cannot be split, return false. What’s your solution’s complexity?

What’s the computational complexity of finding a document’s most frequently used words?
If you’re given 10 TBs of unstructured customer data, how would you go about finding extracting valuable information from it?

How would you ‘disjoin’ two arrays (like JOIN for SQL, but the opposite)?
Create a function that does addition where the numbers are represented as two linked lists.
Create a function that calculates matrix sums.
How would you use Python to read a very large tab-delimited file of numbers to count the frequency of each number?

Write a function that takes a sentence and prints out the same sentence with each word backwards in O(n) time.
Write a function that takes an array, splits the array into every possible set of two arrays, and prints out the max differences between the two array’s minima in O(n) time.
Write a program that does merge sort.
SQL Questions

(Data Analyst) Define and explain the differences between clustered and non-clustered indexes.
(Data Analyst) What are the different ways to return the rowcount of a table?

If you’re given a raw data table, how would perform ETL (Extract, Transform, Load) with SQL to obtain the data in a desired format?
How would you write a SQL query to compute a frequency table of a certain attribute involving two joins? What changes would you need to make if you want to ORDER BY or GROUP BY some attribute? What would you do to account for NULLS?
Brain Teasers and Word Problems

Suppose you have ten bags of marbles with ten marbles in each bag. If one bag weighs differently than the other bags, and you could only perform a single weighing, how would you figure out which one is different?

You are about to hop on a plane to Seattle and want to know if you should carry an umbrella. You call three friends of yours that live in Seattle and ask each, independently, if it’s raining.
Each of your friends will tell you the truth ⅔ of the time and mess with you by lying ⅓ of the time. If all three friends answer “Yes, it’s raining,” what is the probability that is it actually raining in Seattle?
Imagine there are three ants in each corner of an equilateral triangle, and each ant randomly picks a direction and starts traversing the edge of the triangle. What’s the probability that none of the ants collide? What about if there are N ants sitting in N corners of an equilateral polygon?
How many trailing zeros are in 100 factorial (i.e. 100!)?

Imagine you are working with a hospital. Patients arrive at the hospital in a Poisson Distribution, and the doctors attend to the patients in a Uniform Distribution. Write a function or code block that outputs the patient’s average wait time and total number of patients that are attended to by doctors on a random day.

Imagine you’re climbing a staircase that contains n stairs, and you can take any number k steps. How many distinct ways can you reach the top of the staircase? (This is a modification of the original stair step problem)


##### Source:
* https://www.learndatasci.com/data-science-interview-questions/
* https://www.learndatasci.com/free-data-science-books/
* https://developers.google.com/machine-learning/crash-course/prereqs-and-prework
