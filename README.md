# Machine Learning and Deep Learning - Human Resources

<div>
  
In this case study, we will analyze the company's layoffs and the possible relationship between these numbers and other variables analyzed by the company's database. The intention is to avoid losing good employees and keep the best employees in the company. This is an interesting application of data analysis and artificial intelligence algorithms in the human resources area. Studies indicate that the hiring area of ​​a company is a complex and costly task, in addition to wasting resources and time on tasks within the company that do not produce or generate revenue. This financial issue always raises deep questions to be analyzed, and today machine learning and deep learning algorithms can be extremely useful by optimizing time and saving costs through prediction.
</div>

</b>
</b>

<div>
  The database for this is study comes from https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
</div>

</b>
</b>

## SkLearn

<div>
  For Data Analysis, the pandas library is extremely important, from simple tasks like importing the database into your code, to manipulating its elements.Another library that allows us to manipulate, preprocess and use some machine learning algorithms is SkLearn. 
  SkLearn: https://scikit-learn.org/stable/
  
</b>
</b>
  
  Sklearn provides all the documentation needed to apply the algorithms. It is extremely important to understand how to apply its functions, since these codes apply mathematics and statistics. Analyzing a structure, such as a table, requires prior study of the area from which the data was collected, as well as the statistical tools that allow us to extract values ​​from these attributes and analyze them mathematically. Values ​​such as mean, median, mode, standard deviation, and quadratic standard deviation allow us to numerically infer how the data is distributed, possible noise values ​​(outliers), the correlation between each variable, and also use the correct computational tools. For example, poorly treated outliers will influence the mean and distort future analyses; variables with low linear correlation will probably indicate a weak relationship between the variables and, if confirmed through statistical tests and normal distribution, rule out the use of a linear regression algorithm for prediction. These inferences are reflected in choices not only of algorithms, but also in assessing the quality of the data. A poorly distributed database, with little diversity, will result in an algorithm with low accuracy and poor metrics.
</div>
</b>
</b>

## Matplotlib and Seaborn
<div>
  Graphic resources are always great allies for data analysis. I used resouces from Matplotlib to show the histogram from the variables.
A histogram is a bar chart that represents the frequency distribution of numerical data by dividing the data into intervals and showing the frequency of each interval. It is a visual tool for understanding the distribution, patterns, and possible anomalies in a data set. So, I always this graphics in my analysis, as well as Seaborn, which has great features for heat maps and boxplot ( for outliers analysis).
</div>
</b>
</b>
<div>
  For any application of Machine Learning algorithms, such as Neural Networks, the database needs to be evaluable. Bad data will not produce good results, and we need to interpret it so that we can apply corrections to the algorithms. Understand what type of method (classification, regression, supervised or unsupervised learning, correct categorization of attributes, possible outliers, etc.) and use the possible ones, seeking the best accuracy. The metrics used to evaluate the algorithms will also depend on the understanding of the database and the questions needed to be answered. Therefore, F1 score, precision score, mean squared error, recall or accuracy must be very well evaluated.
</b>
  Some of the columns must be removed, in this case, because they don't are useful. The need for a column and its deletion must be carefully evaluated. Nominal attributes must be convert to discrete values (SVM, for example, this is not necessary). 
  </b>
  
  ### One Hot Encoding
  
  One-hot encoding is a technique used in machine learning and data preprocessing to convert categorical data into a numerical format that algorithms can understand. It represents each category as a binary vector, where only one element is "hot" (set to 1), and the rest are 0. This ensures that no ordinal relationship is implied between categories. 
  </b>

  ### MinMaxScaler
  
  The MinMaxScaler is a normalization technique used to scale data within a specific range, usually between 0 and 1. This technique is especially useful in machine learning algorithms that are sensitive to the scale of the data, such as neural networks and distance-based methods like k-nearest neighbors (k-NN).
  </b>

  </b>
  After this process, we can separate the values that will be use to train the algorithm and the values that will test this code. Now, we can use the choosen methods for this case to predict the Attriction values (0 or 1).
  
</div>
</b>
</b>

## Logistic Regression
<div>

  Logistic Regression is a machine learning algorithm used to predict the probability of an event occurring based on a set of independent variables. This algorithm is particularly useful in binary classification problems, where the dependent variable is dichotomous, meaning it has only two classes: 'yes' or 'no'. The logistic function, or sigmoid, is applied to transform the values of the independent variables into a probability between 0 and 1. Logistic regression is widely used in various fields, such as marketing, finance, engineering, and social sciences, to predict events such as customer churn, disease diagnosis, and fraud detection.
  
</div>

</b>
</b>

## Random Forest
<div>
  Random Forest is a machine learning algorithm based on decision trees. It combines multiple decision trees to improve accuracy and reduce the risk of overfitting. It works well with noisy data and does not require much preprocessing. It can be used for classification and regression. The combination of multiple trees reduces the risk of overfitting.
</div>

</b>
</b>

## Artificial Neural Networks 
<div>
  A neural network is a type of machine learning that is modeled after the human brain. This creates an artificial neural network that, through an algorithm, allows the computer to learn by incorporating new data.
  
  The structure of the algorithm is basically: input layer, hidden layers, and output layer. Where between each connection between the neurons of the hidden layer and, subsequently, there are weights. Its connections and the calculation of the weights, the presence of Bias, and its activation and output functions, error function, and propagation will depend on the structure of the type of neural network.
  I'd like to separate a theme just to discuss this algorithms.
</div>

</b>
</b>

## Some Metrics

### Accuracy Score

<div>
  Accuracy_score: Accuracy is a metric used to evaluate the performance of a machine learning model. It is calculated by dividing the number of correct predictions by the total number of predictions. The mathematical formula is: Accuracy = (True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives). This metric is useful for understanding the proportion of correct predictions relative to the total predictions, and is especially important in balanced classification problems.
</div>
  
</b>
</b>

### Mean Squared Error
<div>
  Mean Squared Error:  It calculates the average of the squares of the errors, which are the differences between the predicted and actual values. The MSE is always non-negative, and values closer to zero indicate better model performance.
</div>

</b>
</b>

<div>
  F1_score: The F1 score is a metric used to evaluate the performance of a classification model, especially when dealing with imbalanced datasets. It is the harmonic mean of precision and recall, providing a single score that balances both metrics.
  
</div>
</b>

### Precision Score
<div>
  Precision_score: The precision score is a performance metric used to evaluate the accuracy of a model's predictions. It is defined as the ratio of true positives to the total number of positives predicted by the model. The formula for precision is: [ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} ] This metric is particularly useful in scenarios where the cost of false positives is high, such as spam detection or email filtering. A precision score of 1.0 indicates that all predictions are correct, while a score of 0.5 suggests that half of the predictions are correct.

</div>

</b>

### Recall Score

<div>
</b>
  Recall_score: The recall score is a performance metric used in machine learning and pattern recognition to measure the ability of a model to identify relevant instances. It is defined as the fraction of relevant instances that were correctly identified by the model. In the context of a binary classification problem, recall is calculated as the ratio of true positives (TP) to the total number of relevant instances (TP + false negatives, FN)
</div>
</b>

### True Positive
<div>
  True positive: In a binary decision, if the attribute was classified correctly as 1 (yes), the algorithm classified it correctly, so it's a True positive.
</div>

</b>

### True Negative

<div>
  True negative: In a binary decision, if the attribute was classified correctly as 0 (no), the algorithm classified it correctly, so it's a True positive.

</div>
</b>
</b>


### False Positive
<div>
  False positive: In a binary decision, if the attribute was classified  as 1 (yes), but it should have been classified as 0 (no), the algorithm did not classify it correcty, so it's a False positive.

</div>
</b>

### False Negative
<div>
  False negative:  In a binary decision, if the atribute was classified  as 0 (no), but it should have been classified as 1 (yes), the algorithm did not classify it correcty, so it's a False negative.
</div>

</b>
</b>

## Finally

<div>
  As we can see, the matter of the metric will depends of the database. For example:  if you are building an algorithm that categorizes age groups for movie permissions, you should avoid false positives. A false positive means that you will be allowing adult content for a child, so your algorithm should prioritize the metric Precision as an assessment of the quality of your algorithm.
  The hyperparameters should be tested until the best possible score is reached.
</div>
