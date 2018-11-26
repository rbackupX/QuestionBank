Interview Questions for Python Data Science /Machine Learning
Few pointers for interviews in general before you dive in for the questions.
• Interviews are not exams. You are not supposed to know and you’ll not know everything asked. Its ok to
not know. Don't panic
• In continuation of above, it will rarely happen that you have absolutely no idea about things being asked .
You might have heard about it but may have never explored fully to be expert enough to answer the
problem posed. Develop that opportunity into a genuine discussion, show interest.
• As a beginner in the field your hunger and ability to learn are as important as the things that you have
learnt. Do try to bring out these qualities
• Know your resume well. Fumbling about the things that you mention on your cv could be a red flag for a
lot of interviewers
• Think hard about the specific questions which might be asked to you. Such as , reason for gaps in
employment, reason for your frequent change in jobs , reason for changing domains etc. Be ready with
the answers . Try to make it sound like ‘not’ practiced .
• Be genuine about your technical weak points , don't fake it. Talk about what you are doing to improve on
the same.
• Read up on the domain specifics of the company/role that you are interviewing for. Be prepared and be
curious
• Do ask questions about your roles and responsibilities and discuss them. You don't have to turn down the
opportunity in the interview itself if you don't like the role. This can always be done later after proper
researching about the role at your end.
• Its ok to screw up interviews , its not the end of the world . Treat it as a learning opportunity . Revisit your
mistakes. [Realisation and acceptance of mistake is very important for improvement]. Move on and
prepare for next.
• For the things that you don't know about completely ; answer as much as you know well and do mention
the things that you are not sure about. It reflects sincerity and awareness of one’s areas of improvement .
• Don’t worry about syntax related eccentric behaviour of a programming language . Focus on logic of step.
In this age of google and stackExchange , rarely anyone cares about you mugging up syntax oddities .
Although a basic familiarity is expected which comes with practice .
Ok time for the questions . I haven't put in answers here for the reason that people tend to mug up.
Develop your own style of answers. If you want to discuss on forum, please feel free to do so. Also ; not
having a direct answer here to mug up from will encourage you to explore and learning more than just
what the questions asks.
This list is , by no means exhaustive; But i have tried to cover most of the standard questions here. If you
feel something should be added to the list ; please reach out to us for discussion. Also there is no ‘order’ to
the questions listed . Here we go :
1. How to remove duplicate observations from a data frame in python?
2. How are NAs stored in pandas data frames , how to impute/remove them?
3. What is the difference between numpy arrays and nested lists. Why should we prefer numpy arrays for
data?
4. How to get group wise summary in pandas. give examples of aggregate function.
5. What is the purpose of function ‘copy’. Why is it needed ?
6. What is the difference between tuples, list and sets?
7. How can you randomise contents of a list in python?
8. What is the difference between pass and break in python?
9. Explain what is regularisation and why it is useful?
10. Why is model validation important and what are ways to do that?
11. Why is cross validation better than simple train/test split?
12. What are precision and recall and how do they relate to F_beta/F1 score?
13. What is the difference between F_beta score and KS?
14. What are other performance measures for regression problem than MSE/RMSE
15. Whats is stratified sampling , stratified Kfold ?
16. What is overfitting and what are the ways to control it?
17. Which algorithm will you use for anomaly detection ?
18. Explain what an ROC curve is?
19. What is the difference between RanomForest Models and ExtraTrees?
20. What is the difference bagging and boosting . Give example of each class of models.
21. What can compensate for absence of coefficient in tree models for understanding relationship between response and predictors .
22. What is the difference between non linear pattern captured by SVM and tree based models.
23. What is the difference between lasso and ridge regressions?
24. How is PCA used for dimensionality reduction? How is it different from Factor analysis?
25. What are other distance measures apart from simple euclidian distances ?
26. What impurity measures are used to build decision tree and related models in python’s scimitar learn library ?
27. What is gradient descent method? What is stochastic gradient descent method ?
28. What are different classification metrics that you know about?
29. How do you handle missing values in the data?
30. How to use categorical data in ML algorithms ?
31. Which algorithms to chose for text data : Naive Bayes , Logistic Regression?
32. Is accuracy as a stand alone measure of classification model’s performance good? Discuss the case when it isn't .
33. What is the naive assumption in Naive Bayes?
34. What is difference between KNN and KMeans?
35. What are the limitations of K-means? How is DBSCAN different from K-Means?
36. What are different measures of goodness of cluster fit?
37. What is variable importance in Random Forest model?
38. How are regression trees built ?
39. What is the difference between grid search and random grid search?
40. What is the difference between parameters and hyper parameters ?
41. What is kernel trick in svm?
42. What all hyper parameters can we tune in xgboost ? what do they mean?
43. When can we replace a categorical variable with simple numbers?
44. What is the difference between machine learning and classical statistical models?
45. Walk me through the coding/development process of the last ML project that you worked on. Also give reasons for each step.
46. How would generate a list of random integers in python between a certain range?
47. How do you handle date time data in python?
48. What is cosine similarity ?
49. What are the algorithms that you want to learn further ? why, what are their usage?
50. What do you like/dislike about working with ML algos and data using python?
51. What is the ultimate goal of using Machine Learning Algorithm in business? When will you chose a simple algorithm over a slightly better performing but complex one.
52. What are the challenges with using BeautifulSoup for parsing webdata?
53. What is multicollinearity ? How to counter it ?
Estimation Type Case Studies
Few Pointers here as well:
• There is no right wrong answer to estimation cases
• What is being assessed is the way you think about all possible scenarios , how you can discard irrelevant details using reasonable assumptions and focus on core issues of the problem.
• Focus is on your logical approach to the problem .
• Clarify the question, make sure you and the interviewer are on the same page on every assumption
• Break the problem into smaller pieces if required.
• Discuss your thought process continuously with the interviewer . They are not expecting a 5 minute
silence and a number as answer at the end.
• Write down your assumptions as you go along .
• Be open to changing your assumptions as you discuss them with the interviewer.
• Once you arrive at the answer , do discuss how you can make it more precise given more resources for
data collection.
Now to the Case Studies / Logical questions :
1. A building has 100 floors. Given 2 identical eggs, how can you use them to find the threshold floor? The
egg will break from any particular floor above floor N, including floor N itself.
2. In a given day, how many birthday posts occur on Facebook?
3. You are at a Casino. You have two dices to play with. You win $10 every time you roll a 5. If you play till
you win and then stop, what is the expected pay-out?
4. How many big Macs does McDonald sell every year in US?
5. You are about to get on a plane to Seattle, you want to know whether you have to bring an umbrella or
not. You call three of your random friends and as each one of them if it’s raining. The probability that
your friend is telling the truth is 2/3 and the probability that they are playing a prank on you by lying is
1/3. If all 3 of them tell that it is raining, then what is the probability that it is actually raining in Seattle.
6. You can roll a dice three times. You will be given $X where X is the highest roll you get. You can choose
to stop rolling at any time (example, if you roll a 6 on the first roll, you can stop). What is your expected
pay-out?
7. How can bogus Facebook accounts be detected?
8. How many dentists are there in US?
9. How will you test that there is increased probability of a user to stay active after 6 months given that a
user has more friends now?
10. How many people are using Facebook in California at 1.30 PM on Monday?
11. Can i hit a satellite with a cricket ball? What are the problems that i will encounter ?
12. With a new high speed road coming up between Lucknow and Kanpur , how much the demand of
bread will increase in that region ?
13. How many more ATMs are needed in Mumbai ?
