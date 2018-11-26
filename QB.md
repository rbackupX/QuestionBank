Questions And Answers
 

You are given a train data set having 1000 columns and 1 million rows. The data set is based on a classification problem. Your manager has asked you to reduce the dimension of this data so that model computation time can be reduced. Your machine has memory constraints. What would you do? (You are free to make practical assumptions.)[Reference: https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/]
Is rotation necessary in PCA? If yes, Why? What will happen if you don’t rotate the components?[Reference :https://www.quora.com/Is-rotation-necessary-in-PCA-If-yes-why-What-will-happen-if-you-don%E2%80%99t-rotate-the-components]
You are given a data set. The data set has missing values which spread along 1 standard deviation from the median. What percentage of data would remain unaffected? Why?[Reference :https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/]
You are given a data set on cancer detection. You’ve build a classification model and achieved an accuracy of 96%. Why shouldn’t you be happy with your model performance? What can you do about it?[Reference:https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/]
You are working on a time series data set. You manager has asked you to build a high accuracy model. You start with the decision tree algorithm, since you know it works fairly well on all kinds of data. Later, you tried a time series regression model and got higher accuracy than decision tree model. Can this happen? Why?[https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/]
You are assigned a new project which involves helping a food delivery company save more money. The problem is, company’s delivery team aren’t able to deliver food on time. As a result, their customers get unhappy. And, to keep them happy, they end up delivering food for free. Which machine learning algorithm can save them?
You came to know that your model is suffering from low bias and high variance. Which algorithm should you use to tackle it? Why?
You are given a data set. The data set contains many variables, some of which are highly correlated and you know about it. Your manager has asked you to run PCA. Would you remove correlated variables first? Why?
After spending several hours, you are now anxious to build a high accuracy model. As a result, you build 5 GBM models, thinking a boosting algorithm would do the magic. Unfortunately, neither of models could perform better than benchmark score. Finally, you decided to combine those models. Though, ensembled .
Running a binary classification tree algorithm is the easy part. Do you know how does a tree splitting takes place i.e. how does the tree decide which variable to split at the root node and succeeding nodes?
You’ve built a random forest model with 10000 trees. You got delighted after getting training error as 0.00. But, the validation error is 34.23. What is going on? Haven’t you trained your model perfectly?
You’ve got a data set to work having p (no. of variable) > n (no. of observation). Why is OLS as bad option to work with? Which techniques would be best to use? Why?
You have built a multiple regression model. Your model R² isn’t as good as you wanted. For improvement, your remove the intercept term, your model R² becomes 0.8 from 0.3. Is it possible? How?
After analyzing the model, your manager has informed that your regression model is suffering from multicollinearity. How would you check if he’s true? Without losing any information, can you still build a better model?
You are given a data set consisting of variables having more than 30% missing values? Let’s say, out of 50 variables, 8 variables have missing values higher than 30%. How will you deal with them?
‘People who bought this, also bought…’ recommendations seen on amazon is a result of which algorithm?
Which data visualisation libraries do you use? What are your thoughts on the best data visualisation tools?
How would you implement a recommendation system for our company’s users?
How can we use your machine learning skills to generate revenue?
What are the last machine learning papers you’ve read?
Do you have research experience in machine learning?
What are your favorite use cases of machine learning models?
How would you approach the “Netflix Prize” competition?
Where do you usually source datasets?
How do you think Google is training data for self-driving cars?
What do you understand by Type I vs Type II error ?
You are working on a classification problem. For validation purposes, you’ve randomly sampled the training data set into train and validation. You are confident that your model will work incredibly well on unseen data since your validation accuracy i
State the universal approximation theorem? What is the technique used to prove that?
Given the universal approximation theorem, why can’t a MLP still reach a arbitrarily small positive error?
What is the mathematical motivation of Deep Learning as opposed to standard Machine Learning techniques?
In standard Machine Learning vs. Deep Learning, how is the order of number of samples related to the order of regions that can be recognized in the function space?
What are the reasons for choosing a deep model as opposed to shallow model? (1. Number of regions O(2^k) vs O(k) where k is the number of training examples 2. # linear regions carved out in the function space depends exponentially on the depth. )
How Deep Learning tackles the curse of dimensionality?(Other sources(https://www.analyticsvidhya.com/blog/2017/01/must-know-questions-deep-learning)
How will you implement dropout during forward and backward pass?
What do you do if Neural network training loss/testing loss stays constant? (ask if there could be an error in your code, going deeper, going simpler…)
Why do RNNs have a tendency to suffer from exploding/vanishing gradient? How to prevent this? (Talk about LSTM cell which helps the gradient from vanishing, but make sure you know why it does so. Talk about gradient clipping, and discuss whether to clip the gradient element wise, or clip the norm of the gradient.)
Do you know GAN, VAE, and memory augmented neural network? Can you talk about it?
Does using full batch means that the convergence is always better given unlimited power? (Beautiful explanation by Alex Seewald: https://www.quora.com/Is-full-batch-gradient-descent-with-unlimited-computer-power-always-better-than-mini-batch-gradient-descent)
What is the problem with sigmoid during backpropagation? (Very small, between 0.25 and zero.)
Given a black box machine learning algorithm that you can’t modify, how could you improve its error? (you can transform the input for example.)
How to find the best hyper parameters? (Random search, grid search, Bayesian search (and what it is?))
What is transfer learning?
Compare and contrast L1-loss vs. L2-loss and L1-regularization vs. L2-regularization.
Can you state Tom Mitchell’s definition of learning and discuss T, P and E?
What can be different types of tasks encountered in Machine Learning?
What are supervised, unsupervised, semi-supervised, self-supervised, multi-instance learning, and reinforcement learning?
Loosely how can supervised learning be converted into unsupervised learning and vice-versa?
Consider linear regression. What are T, P and E?
Derive the normal equation for linear regression.
What do you mean by affine transformation? Discuss affine vs. linear transformation.
Discuss training error, test error, generalization error, overfitting, and underfitting.
Compare representational capacity vs. effective capacity of a model.
Discuss VC dimension.
What are nonparametric models? What is nonparametric learning?
What is an ideal model? What is Bayes error? What is/are the source(s) of Bayes error occur?
What is the no free lunch theorem in connection to Machine Learning?
What is regularization? Intuitively, what does regularization do during the optimization procedure? (expresses preferences to certain solutions, implicitly and explicitly)
What is weight decay? What is it added?
What is a hyperparameter? How do you choose which settings are going to be hyperparameters and which are going to be learnt? (either difficult to optimize or not appropriate to learn – learning model capacity by learning the degree of a polynomial or coefficient of the weight decay term always results in choosing the largest capacity until it overfits on the training set)
Why is a validation set necessary?
What are the different types of cross-validation? When do you use which one?
What are point estimation and function estimation in the context of Machine Learning? What is the relation between them?
What is the maximal likelihood of a parameter vector $theta$? Where does the log come from?
Prove that for linear regression MSE can be derived from maximal likelihood by proper assumptions.
Why is maximal likelihood the preferred estimator in ML? (consistency and efficiency)
Under what conditions do the maximal likelihood estimator guarantee consistency?
What is cross-entropy of loss? (trick question)
What is the difference between an optimization problem and a Machine Learning problem?
How can a learning problem be converted into an optimization problem?
What is empirical risk minimization? Why the term empirical? Why do we rarely use it in the context of deep learning?
Name some typical loss functions used for regression. Compare and contrast. (L2-loss, L1-loss, and Huber loss)
What is the 0-1 loss function? Why can’t the 0-1 loss function or classification error be used as a loss function for optimizing a deep neural network? (Non-convex, gradient is either 0 or undefined. 
1.What’s the difference between a generative and discriminative model?
When should you use classification over regression?
What evaluation approaches would you work to gauge the effectiveness of a machine learning model?
models are known to return high accuracy, but you are unfortunate. Where did you miss?
When is Ridge regression favorable over Lasso regression?
While working on a data set, how do you select important variables? Explain your methods.
We know that one hot encoding increasing the dimensionality of a data set. But, label encoding doesn’t. How ?
Explain machine learning to me like a 5 year old.
Considering the long list of machine learning algorithm, given a data set, how do you decide which one to use?
Do you suggest that treating a categorical variable as continuous variable would result in a better predictive model?
When does regularization becomes necessary in Machine Learning?
What are parametric models? Give an example?
What are 3 data preprocessing techniques to handle outliers?
What are 3 ways of reducing dimensionality?
How much data should you allocate for your training, validation, and test sets?
If you split your data into train/test splits, is it still possible to overfit your model?
How can you choose a classifier based on training set size?
Explain Latent Dirichlet Allocation (LDA)
What are some key business metrics for (S-a-a-S startup | Retail bank | e-Commerce site)?
How can you help our marketing team be more efficient?
Differentiate between Data Science , Machine Learning and AI.((https://www.dezyre.com/article/100-data-science-interview-questions-and-answers-general-for-2018/184))
Python or R – Which one would you prefer for text analytics?
Which technique is used to predict categorical responses?
What is Interpolation and Extrapolation?
What is power analysis?
What is the difference between Supervised Learning and Unsupervised Learning?
Explain the use of Combinatorics in data science.
Why is vectorization considered a powerful method for optimizing numerical code?
What is the goal of A/B Testing?
What are various steps involved in an analytics project?
Can you use machine learning for time series analysis?
What is the difference between Bayesian Estimate and Maximum Likelihood Estimation (MLE)?
What is multicollinearity and how you can overcome it?
What is the difference between squared error and absolute error?
Differentiate between wide and tall data formats?
How would you develop a model to identify plagiarism?
You created a predictive model of a quantitative outcome variable using multiple regressions. What are the steps you would follow to validate the model?
What do you understand by long and wide data formats?
What is the importance of having a selection bias?
What do you understand by Fuzzy merging ? Which language will you use to handle it?
How can you deal with different types of seasonality in time series modelling?
What makes a dataset gold standard?
Can you write the formula to calculate R-square?
Difference between Generative and Discriminative models.
How will you assess the statistical significance of an insight whether it is a real insight or just by chance?
How would you create a taxonomy to identify key customer trends in unstructured data?
What do you understand by feature vectors?
How do data management procedures like missing data handling make selection bias worse?
How’s EM done?
How can you plot ROC curves for multiple classes. – There is something called as amacro-averaging of weights where PRE = (PRE1 + PRE2 + — + PREk )/K49.Text methods (latent, etc), he asked if I knew anything about these.
What is the difference between inductive machine learning and deductive machine learning?
How will you know which machine learning algorithm to choose for your classification problem?
What are Bayesian Networks (BN) ?
What is algorithm independent machine learning?
What is classifier in machine learning?
In what areas Pattern Recognition is used?
What is Genetic Programming?
What is Inductive Logic Programming in Machine Learning?
What is inductive machine learning?
What are the five popular algorithms of Machine Learning?
What are the different Algorithm techniques in Machine Learning?
List down various approaches for machine learning?
What are the different methods for Sequential Supervised Learning?
What is batch statistical learning?
What is PAC Learning?
What is sequence learning?
What are two techniques of Machine Learning ?
How to use labeled and unlabeled data?
What if you don’t have any labeled data?
What if your data set is skewed (e.g. 99.99 % positive and 0.01% negative labels)?
How to make training faster?
How to make predictions faster?
Write the equation describing a dynamical system. Can you unfold it? Now, can you use this to describe a RNN? (include hidden, input, output, etc.)
What determines the size of an unfolded graph?
What are the advantages of an unfolded graph? (arbitrary sequence length, parameter sharing, and illustrate information flow during forward and backward pass)
What does the output of the hidden layer of a RNN at any arbitrary time t represent?
Are the output of hidden layers of RNNs lossless? If not, why?
RNNs are used for various tasks. From a RNNs point of view, what tasks are more demanding than others?
Discuss some examples of important design patterns of classical RNNs.
Write the equations for a classical RNN where hidden layer has recurrence. How would you define the loss in this case? What problems you might face while training it? (Discuss runtime)
What is backpropagation through time? (BPTT)
Consider a RNN that has only output to hidden layer recurrence. What are its advantages or disadvantages compared to a RNNhaving only hidden to hidden recurrence?
What is Teacher forcing? Compare and contrast with BPTT.
What is the disadvantage of using a strict teacher forcing technique? How to solve this?
Explain the vanishing/exploding gradient phenomenon for recurrent neural networks. (use scalar and vector input scenarios)
Why don’t we see the vanishing/exploding gradient phenomenon in feedforward networks? (weights are different in different layers – Random block intialization paper)
What is the key difference in architecture of LSTMs/GRUs compared to traditional RNNs? (Additive update instead of multiplicative)
What is the difference between LSTM and GRU?
Explain Gradient Clipping.
Adam and RMSProp adjust the size of gradients based on previously seen gradients. Do they inherently perform gradient clipping? If no, why?
Discuss RNNs in the context of Bayesian Machine Learning.
Can we do Batch Normalization in RNNs? If not, what is the alternative? (BNorm would need future data; Layer Norm)
What is an Autoencoder? What does it “auto-encode”?
What were Autoencoders traditionally used for? Why there has been a resurgence of Autoencoders for generative modeling?
What is recirculation?
What loss functions are used for Autoencoders?
What is a linear autoencoder? Can it be optimal (lowest training reconstruction error)? If yes, under what conditions?
What is the difference between Autoencoders and PCA (can also be used for reconstruction – https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com).
What is the impact of the size of the hidden layer in Autoencoders?
What is an undercomplete Autoencoder? Why is it typically used for?
What is a linear Autoencoder? Discuss it’s equivalence with PCA. (only valid for undercomplete) Which one is better in reconstruction?
What problems might a nonlinear undercomplete Autoencoder face?
What are overcomplete Autoencoders? What problems might they face? Does the scenario change for linear overcomplete autoencoders? (identity function)
Discuss the importance of regularization in the context of Autoencoders.
Why does generative autoencoders not require regularization?
What are sparse autoencoders?
What is a denoising autoencoder? What are its advantages? How does it solve the overcomplete problem?
What is score matching? Discuss it’s connections to DAEs.
Are there any connections between Autoencoders and RBMs?
What is manifold learning? How are denoising and contractive autoencoders equipped to do manifold learning?
What is a contractive autoencoder? Discuss its advantages. How does it solve the overcomplete problem?
Why is a contractive autoencoder named so? (intuitive and mathematical)
What are the practical issues with CAEs? How to tackle them?
What is a stacked autoencoder? What is a deep autoencoder? Compare and contrast.
Compare the reconstruction quality of a deep autoencoder vs. PCA.
What is predictive sparse decomposition?
Discuss some applications of Autoencoders.
What is representation learning? Why is it useful? (for a particular architecture, for other tasks, etc.)
What is the relation between Representation Learning and Deep Learning?
What is one-shot and zero-shot learning (Google’s NMT)? Give examples.
What trade offs does representation learning have to consider?
What is greedy layer-wise unsupervised pretraining (GLUP)? Why greedy? Why layer-wise? Why unsupervised? Why pretraining?
What were/are the purposes of the above technique? (deep learning problem and initialization)
Why does unsupervised pretraining work?
When does unsupervised training work? Under which circumstances?
Why might unsupervised pretraining act as a regularizer?
What is the disadvantage of unsupervised pretraining compared to other forms of unsupervised learning?
How do you control the regularizing effect of unsupervised pre-training?
How to select the hyperparameters of each stage of GLUP?
What cross-validation technique would you use on a time series dataset?(Time series data )
How would you handle an imbalanced dataset?(Classification Algo in various situations)
Name an example where ensemble techniques might be useful.(Ensemble models)
What’s the “kernel trick” and how is it useful?(SVM)((https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/)
Both being tree based algorithm, how is random forest different from Gradient boosting algorithm (GBM)?
What is convex hull ? (svm)
What are the advantages and disadvantages of decision trees?(DT)
What are the advantages and disadvantages of neural networks?(Deep Learning)
Why are ensemble methods superior to individual models?(Ensemble Models)
Explain bagging.(NLP)
What are Recommender Systems?(Recommendation system)
Why data cleaning plays a vital role in analysis?(EDA)
Differentiate between univariate, bivariate and multivariate analysis.(EDA)
What is Linear Regression?
What is Collaborative filtering?(recommendation systems)
Are expected value and mean value different?
What are categorical variables?(classification algo)
How can you iterate over a list and also retrieve element indices at the same time?(Python)
During analysis, how do you treat missing values?(EDA)
Write a function that takes in two sorted lists and outputs a sorted list that is their union. (Python)
How are confidence intervals constructed and how will you interpret them?(Probability )
How will you explain logistic regression to an economist, physican scientist and biologist?(Logistic regression)
Is it better to have too many false negatives or too many false positives?(Performance measurement models)
What do you understand by statistical power of sensitivity and how do you calculate it?(Probability)
Give some situations where you will use an SVM over a RandomForest Machine Learning algorithm and vice-versa.(SVM)
Write a program in Python which takes input as the diameter of a coin and weight of the coin and produces output as the money value of the coin.(Programming)
What are the basic assumptions to be made for linear regression?(Linear regression)
Difference between convex and non-convex cost function; what does it mean when a cost function is non-convex? (SVM)
Stochastic Gradient Descent: if it is faster, why don’t we always use it?(Linear regression)
Difference between SVM and Log R – Easy(SVM)
Does SVM give any probabilistic output – I said no it doesn’t and it was wrong! He gave me hints but I couldn’t figure it out!(SVM)
What are the support vectors in SVM
Mention the difference between Data Mining and Machine learning?(General)
You are given a data set. The data set has missing values which spread along 1 standard deviation from the median. What percentage of data would remain unaffected? Why?(EDA)
Why is Naïve Bayes machine learning algorithm naïve?(Naive Bayes)
Explain prior probability, likelihood and marginal likelihood in context of naïve Bayes algorithm?
What are the three stages to build the hypotheses or model in machine learning?
What is the standard approach to supervised learning?
What is ‘Training set’ and ‘Test set’?
List down various approaches for machine learning?
How to know that your model is suffering from low bias and high variance. Which algorithm should you use to tackle it? Why?
Name some feature extraction techniques used for dimensionality reduction.
List some use cases where classification machine learning algorithms can be used.
What kind of problems does regularization solve?
How much data will you allocate for your training, validation and test sets?
Which one would you prefer to choose – model accuracy or model performance?
Describe some popular machine learning methods.
What is not Machine Learning?
Explain what is the function of ‘Unsupervised Learning’?
How will you differentiate between supervised and unsupervised learning? Give few examples of algorithms for supervised learning?
What is linear regression? Why is it called linear?
How does the variance of the error term change with the number of predictors, in OLS?
Do we always need the intercept term? When do we need it and when do we not?
How interpretable is the given machine learning model?
What will you do if training results in very low accuracy?
Does the developed machine learning model have convergence problems?
Which tools and environments have you used to train and assess machine learning models?
How will you apply machine learning to images?
What is collinearity and what to do with it?
How to remove multicollinearity?
What is overfitting a regression model? What are ways to avoid it?
What is loss function in a Neural Network?
Explain the difference between MLE and MAP inference.
What is boosting?
If the gradient descent does not converge, what could be the problem?
How will you check for a valid binary search tree?
How to check if the regression model fits the data well?
What are parametric models?()
What’s the trade-off between bias and variance?
Explain how a ROC curve works.(Performance Measurement Models)
What is the Box-Cox transformation used for?(Probability)
Define precision and recall.(Performance measurement models?
what is the function of ‘Unsupervised Learning’?(Unsuperwised learning)
What is Perceptron in Machine Learning?(Deep Learning)
What is ensemble learning?(Ensemble Models)
What are the two paradigms of ensemble methods?(Ensemble Models)
What is PCA, KPCA and ICA used for?
You are given a data set. The data set has missing values which spread along 1 standard deviation from the median. What percentage of data would remain unaffected? Why?(https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/)
You are given a data set consisting of variables having more than 30% missing values? Let’s say, out of 50 variables, 8 variables have missing values higher than 30%. How will you deal with them?(https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/)
Compare “Frequentist probability” vs. “Bayesian probability”?
What is a random variable?
What is a joint probability distribution?
What are the conditions for a function to be a probability mass function?
What are the conditions for a function to be a probability density function 
What is a marginal probability? Given the joint probability function, how will you calculate it?
What is conditional probability? Given the joint probability function, how will you calculate it?
State the Chain rule of conditional probabilities.
What are the conditions for independence and conditional independence of two random variables?
What are expectation, variance and covariance?
Compare covariance and independence.
What is the covariance for a vector of random variables?
What is a Bernoulli distribution? Calculate the expectation and variance of a random variable that follows Bernoulli distribution?
What is a multinoulli distribution?
What is a normal distribution?
Why is the normal distribution a default choice for a prior over a set of real numbers?
What is the central limit theorem?
What are exponential and Laplace distribution?
What are Dirac distribution and Empirical distribution?
What is mixture of distributions?
Name two common examples of mixture of distributions? (Empirical and Gaussian Mixture)
Is Gaussian mixture model a universal approximator of densities?
Write the formula for logistic and softplus function.
Write the formula for Bayes rule.
What do you mean by measure zero and almost everywhere?
If two random variables are related in a deterministic way, how are the PDFs related?
Define self-information. What are its units?
What are Shannon entropy and differential entropy?
What is Kullback-Leibler (KL) divergence?
Can KL divergence be used as a distance measure?
Define cross-entropy.
What are structured probabilistic models or graphical models?
In the context of structured probabilistic models, what are directed and undirected models? How are they represented? What are cliques in undirected structured probabilistic models?
What is Bayes’ Theorem? How is it useful in a machine learning context?
Why is “Naive” Bayes naive?
What’s a Fourier transform?
What’s the difference between probability and likelihood?
Explain prior probability, likelihood and marginal likelihood in context of naive Bayes algorithm?
What is the difference between covariance and correlation?
Is it possible capture the correlation between continuous and categorical variable? If yes, how?
What is the Box-Cox transformation used for?
What do you understand by the term Normal Distribution?
What does P-value signify about the statistical data?
A test has a true positive rate of 100% and false positive rate of 5%. There is a population with a 1/1000 rate of having the condition the test identifies. Considering a positive test, what is the probability of having that condition?
How you can make data normal using Box-Cox transformation?
Explain about the box cox transformation in regression models.
What is the difference between skewed and uniform distribution?
What do you understand by Hypothesis in the content of Machine Learning?
How will you find the correlation between a categorical variable and a continuous variable ?
What does LogR give ? I said Posterior probability (P(y|x=0 or x=1))
Evaluation of LogR –
How are the params updated – I was able to answer with formulae!
When doing an EM for GMM, how do you find the mixture weights ? I replied that for 2 Gaussians, the prior or the mixture weight can be assumed to be a Bernoulli distribution.
If x ~ N(0,1), what does 2x follow
How would you sample for a GMM
How to sample from a Normal Distribution with known mean and variance.
In experimental design, is it necessary to do randomization? If yes, why
How do you handle missing or corrupted data in a dataset?
Do you have experience with Spark or big data tools for machine learning?
In k-means or kNN, we use euclidean distance to calculate the distance between nearest neighbours. Why not manhattan distance ?(https://www.analyticsvidhya.com/blog/2017/09/30-questions-test-k-nearest-neighbors-algorithm/)
How to test and know whether or not we have overfitting problem?(https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/how-to-determine-overfitting-and-underfitting/)
How is kNN different from k-means clustering?(https://stats.stackexchange.com/questions/56500/what-are-the-main-differences-between-k-means-and-k-nearest-neighbours)
Can you explain the difference between a Test Set and a Validation Set?(https://stackoverflow.com/questions/2976452/whats-is-the-difference-between-train-validation-and-test-set-in-neural-netwo)
How can you avoid overfitting in KNN?(https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/how-to-determine-overfitting-and-underfitting/)
Which is more important to you– model accuracy, or model performance?
Can you cite some examples where a false positive is important than a false negative?
Can you cite some examples where a false negative important than a false positive?
Can you cite some examples where both false positive and false negatives are equally important?
What is the most frequent metric to assess model accuracy for classification problems?
Why is Area Under ROC Curve (AUROC) better than raw accuracy as an out-of- sample evaluation metric?
Define Similarity or Distance matrix.?
Time complexity of Naive Bayes algo  Best and worst cases?
What are the differences between “Bayesian” and “Frequentist” approach for Machine Learning?
Compare and contrast maximum likelihood and maximum a posteriori estimation.
How does Bayesian methods do automatic feature selection?
What do you mean by Bayesian regularization?
When will you use Bayesian methods instead of Frequentist methods? (Small dataset, large feature set)
After analysing the model, your manager has informed that your regression model is suffering from multicollinearity. How would you check if he’s true? Without losing any information, can you still build a better model?(https://google-interview-hacks.blogspot.in/2017/04/after-analyzing-model-your-manager-has.html)
What are the basic assumptions to be made for linear regression?(Refer:https://www.statisticssolutions.com/assumptions-of-linear-regression/)
What is the difference between stochastic gradient descent (SGD) and gradient descent (GD)?(https://stats.stackexchange.com/questions/317675/gradient-descent-gd-vs-stochastic-gradient-descent-sgd)
When would you use GD over SDG, and vice-versa?(https://elitedatascience.com/machine-learning-interview-questions-answers)
How do you decide whether your linear regression model fits the data?(https://www.researchgate.net/post/What_statistical_test_is_required_to_assess_goodness_of_fit_of_a_linear_or_nonlinear_regression_equation)
Is it possible to perform logistic regression with Microsoft Excel?(https://www.youtube.com/watch?v=EKRjDurXau0)
When will you use classification over regression?(https://www.quora.com/When-will-you-use-classification-over-regression)
Why isn’t Logistic Regression called Logistic Classification?(Refer :https://stats.stackexchange.com/questions/127042/why-isnt-logistic-regression-called-logistic-classification/127044)
Give some situations where you will use an SVM over a RandomForest Machine Learning algorithm and vice-versa.(https://datascience.stackexchange.com/questions/6838/when-to-use-random-forest-over-svm-and-vice-versa)
What is convex hull ?(https://en.wikipedia.org/wiki/Convex_hull)
What is a large margin classifier?
Why SVM is an example of a large margin classifier?
SVM being a large margin classifier, is it influenced by outliers? (Yes, if C is large, otherwise not)
What is the role of C in SVM?
In SVM, what is the angle between the decision boundary and theta?
What is the mathematical intuition of a large margin classifier?
What is a kernel in SVM? Why do we use kernels in SVM?
What is a similarity function in SVM? Why it is named so?
How are the landmarks initially chosen in an SVM? How many and where?
Can we apply the kernel trick to logistic regression? Why is it not used in practice then?
What is the difference between logistic regression and SVM without a kernel? (Only in implementation – one is much more efficient and has good optimization packages)
How does the SVM parameter C affect the bias/variance trade off? (Remember C = 1/lambda; lambda increases means variance decreases)
How does the SVM kernel parameter sigma^2 affect the bias/variance trade off?
Can any similarity function be used for SVM? (No, have to satisfy Mercer’s theorem)
Logistic regression vs. SVMs: When to use which one? ( Let’s say n and m are the number of features and training samples respectively. If n is large relative to m use log. Reg. or SVM with linear kernel, If n is small and m is intermediate, SVM with Gaussian kernel, If n is small and m is massive, Create or add more features then use log. Reg. or SVM without a kernel)
What is the difference between supervised and unsupervised machine learning?
You are working on a time series data set. You manager has asked you to build a high accuracy model. You start with the decision tree algorithm, since you know it works fairly well on all kinds of data. Later, you tried a time series regression model and got higher accuracy than decision tree model. Can this happen? Why?(Refer :https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/)
Running a binary classification tree algorithm is the easy part. Do you know how does a tree splitting takes place i.e. how does the tree decide which variable to split at the root node and succeeding nodes?(Refer:https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/)
You’ve built a random forest model with 10000 trees. You got delighted after getting training error as 0.00. But, the validation error is 34.23. What is going on? Haven’t you trained your model perfectly?(Refer : https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/)
How would you implement a recommendation system for our company’s users?(https://www.infoworld.com/article/3241852/machine-learning/how-to-implement-a-recommender-system.html)
How would you approach the “Netflix Prize” competition?(Refer http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/)
‘People who bought this, also bought…’ recommendations seen on amazon is a result of which algorithm?(Please refer Apparel recommendation system case study,Refer:https://measuringu.com/affinity-analysis/)
Pick an algorithm. Write the psuedo-code for a parallel implementation.
What are some differences between a linked list and an array?(Programming)
Describe a hash table.
What is sampled softmax?
Why is it difficult to train a RNN with SGD?
How do you tackle the problem of exploding gradients? (By gradient clipping)
What is the problem of vanishing gradients? (RNN doesn’t tend to remember much things from the past)
How do you tackle the problem of vanishing gradients? (By using LSTM)
Explain the memory cell of a LSTM. (LSTM allows forgetting of data and using long memory when appropriate.)
What type of regularization do one use in LSTM?
What is Beam Search?
How to automatically caption an image? (CNN + LSTM)
What is the mathematical motivation of Deep Learning as opposed to standard Machine Learning techniques?
In standard Machine Learning vs. Deep Learning, how is the order of number of samples related to the order of regions that can be recognized in the function space?
What are the reasons for choosing a deep model as opposed to shallow model? (1. Number of regions O(2^k) vs O(k) where k is the number of training examples 2. # linear regions carved out in the function space depends exponentially on the depth. )
How Deep Learning tackles the curse of dimensionality?
Why do RNNs have a tendency to suffer from exploding/vanishing gradient? How to prevent this? (Talk about LSTM cell which helps the gradient from vanishing, but make sure you know why it does so. Talk about gradient clipping, and discuss whether to clip the gradient element wise, or clip the norm of the gradient.)
What is the problem with sigmoid during backpropagation? (Very small, between 0.25 and zero.)
What is transfer learning?
Write the equation describing a dynamical system. Can you unfold it? Now, can you use this to describe a RNN? (include hidden, input, output, etc.)
What determines the size of an unfolded graph?
What are the advantages of an unfolded graph? (arbitrary sequence length, parameter sharing, and illustrate information flow during forward and backward pass)
What does the output of the hidden layer of a RNN at any arbitrary time t represent?
Are the output of hidden layers of RNNs lossless? If not, why?
RNNs are used for various tasks. From a RNNs point of view, what tasks are more demanding than others?
Discuss some examples of important design patterns of classical RNNs.
Write the equations for a classical RNN where hidden layer has recurrence. How would you define the loss in this case? What problems you might face while training it? (Discuss runtime)
What is backpropagation through time? (BPTT)
Consider a RNN that has only output to hidden layer recurrence. What are its advantages or disadvantages compared to a RNNhaving only hidden to hidden recurrence?
What is Teacher forcing? Compare and contrast with BPTT.
What is the disadvantage of using a strict teacher forcing technique? How to solve this?
Explain the vanishing/exploding gradient phenomenon for recurrent neural networks. (use scalar and vector input scenarios)
Why don’t we see the vanishing/exploding gradient phenomenon in feedforward networks? (weights are different in different layers – Random block intialization paper)
What is the key difference in architecture of LSTMs/GRUs compared to traditional RNNs? (Additive update instead of multiplicative)
What is the difference between LSTM and GRU?
Explain Gradient Clipping.
Adam and RMSProp adjust the size of gradients based on previously seen gradients. Do they inherently perform gradient clipping? If no, why?
Discuss RNNs in the context of Bayesian Machine Learning.
Can we do Batch Normalization in RNNs? If not, what is the alternative? (BNorm would need future data; Layer Norm)
What is representation learning? Why is it useful? (for a particular architecture, for other tasks, etc.)
What is the relation between Representation Learning and Deep Learning?
What is one-shot and zero-shot learning (Google’s NMT)? Give examples.
What trade offs does representation learning have to consider?
What is greedy layer-wise unsupervised pretraining (GLUP)? Why greedy? Why layer-wise? Why unsupervised? Why pretraining?
What were/are the purposes of the above technique? (deep learning problem and initialization)
Why does unsupervised pretraining work?
When does unsupervised training work? Under which circumstances?
Why might unsupervised pretraining act as a regularizer?
What is the disadvantage of unsupervised pretraining compared to other forms of unsupervised learning?
How do you control the regularizing effect of unsupervised pre-training?
How to select the hyperparameters of each stage of GLUP?





https://www.jeremyjordan.me/hyperparameter-tuning/
https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html
https://www.jeremyjordan.me/data-science/
https://www.jeremyjordan.me/imbalanced-data/
https://stats.stackexchange.com/questions/264533/how-should-feature-selection-and-hyperparameter-optimization-be-ordered-in-the-m


External Resources:

1.https://www.analyticsvidhya.com/blog/2017/09/30-questions-test-k-nearest-neighbors-algorithm/
1.https://www.analyticsvidhya.com/blog/2017/08/skilltest-logistic-regression/

2.https://www.listendata.com/2017/03/predictive-modeling-interview-questions.html

3.https://www.analyticsvidhya.com/blog/2017/07/30-questions-to-test-a-data-scientist-on-linear-regression/

4.https://www.analyticsvidhya.com/blog/2016/12/45-questions-to-test-a-data-scientist-on-regression-skill-test-regression-solution/

5. https://www.listendata.com/2018/03/regression-analysis.html