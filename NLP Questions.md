## What are the ways one can encode categorical Features ?
 1. LabelEncoder and OneHotEncoder
 2. DictVectorizer
 3. Pandas get_dummies

### 1. LabelEncoder and OneHotEncoder

##### LabelEncoder
LabelEncoder converts each class under specified feature to a numerical value. 
LabelEncoder is not a good choice for all problems as it brings in a natural ordering for different classes. Which may 
For Example if apples, oranges and bananas are label Encoded we will get 
apples=1
oranges=2
bananas=3
Does it mean bananas is closer to oranges than apples ???
  The answer is obviously no. Thus allowing model learning this result will lead to poor performance. 
  Therefore, for dataframe containing multi class features, a further step of OneHotEncoder is needed. 

##### OneHotEncoder
For each class under a categorical feature, a new column is created for it. 
sparse = False 
sparse = True


