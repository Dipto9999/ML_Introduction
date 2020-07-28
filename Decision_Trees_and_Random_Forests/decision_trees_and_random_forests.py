#############################################
###### Import Open-Software Libraries #######
#############################################

# Import Numerical Computing Libraries
import pandas as pd
import numpy as np
# Import Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib visualizations will embed themselves 
# directly in our Jupyter Notebook. This will make them easier to 
# access and interpret.
%matplotlib inline

#######################################
######## Investigate Data Set #########
#######################################

# Import Data Set into Python Script.
raw_data = pd.read_csv('kyphosis-data.csv')

# Look at features included in Data Set.
    # Kyphosis column contains a value of present or absent depending on whether the individual had the disease.
    # Age column contains the patient's age in months.
    # Number column contains the number of vertebrae involved in operation.
    # Start column describes top-most vertebrae that was operated on.
raw_data.columns

############################################################################################
###### Conduct Exploratory Data Analysis on Data Set (Similar to Logistic Regression) ######
############################################################################################

## Note: Exploratory Data Analysis usually involves calculating
## aggregate data or building visualizations.

# It's important to understand the size of Data Set for Machine Learning Engineers
    # The pandas library method info can be invoked on a DataFrame to let you know the
    # number of observations in the Data Set. (e.g. It should be 81 for our relatively small Data Set.)
raw_data.info()

# Use seaborn library to generate a pairplot and visualize what's happening with each feature.
sns.pairplot(raw_data, hue = 'Kyphosis')

#################################################################
###### Separate Data Set into Training Data and Test Data #######
#################################################################

# Import function train_test_split from scikit-learn's model_selection module
from sklearn.model_selection import train_test_split

# Specify x-values and y-values 
x = raw_data.drop('Kyphosis', axis = 1)
y = raw_data['Kyphosis']

# Use a test size of 30% to train our model. Use list unpacking to assign Training Data and Test Data.
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)

############################################################
#### Train Our Decision Tree Model and Make Predictions ####
############################################################

# Import class DecisionTreeClassifier from tree module of scikit-learn 
from sklearn.tree import DecisionTreeClassifier

# Create an object of class DecisionTreeClassifier
decision_tree_model = DecisionTreeClassifier()

# Invoke method fit on object decision_tree_model by passing in x_training_Data and y_training data variables.
decision_tree_model.fit(x_training_data, y_training_data)

# Call method predict on object decision_tree_model and pass in x_test_data
decision_tree_predictions = decision_tree_model.predict(x_test_data)

###########################################################
######## Evaluate Decision Tree Model Performance #########
###########################################################

# Import scikit-learn's built in functions classification_report and confusion_matrix.
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Generate a Classification Report
print (classification_report(y_test_data, decision_tree_predictions))

# Generate a Confusion Matrix
    # Should notice incorrect predictions on 5 data points:
        # 2 False Positives
        # 3 False Negatives
print(confusion_matrix(y_test_data, decision_tree_predictions))

#############################################
###### Building a Random Forests Model ######
#############################################

# Import class RandomForestClassifier from ensemble module of scikit-learn.
from sklearn.ensemble import RandomForestClassifier

# Create an object of class RandomForestClassifier
random_forest_model = RandomForestClassifier()

## Note: The class RandomForestClassifier has a paramaeter n_estimators 
## which specifies the number of trees in the forest. By default, it is 100.
## However, it may be changed.

# Train the Random Forests Model
random_forest_model.fit(x_training_data, y_training_data)

# Calculate some predictions using our object random_forest_model.
random_forest_predictions = random_forest_model.predict(x_test_data)

######################################################
###### Evaluate Random Forest Model Performance ######
######################################################

# Generate a Classification Report
print(classification_report(y_test_data, random_forest_predictions))
# Generate a Confusion Matrix
    # Notice that the Random Forest Model hasn't performed significantly
    # better than the Decision Tree Model. This is due to the small Data Set.
print(confusion_matrix(y_test_data, random_forest_predictions))

## Note: It is extremely likely for Random Forests to perform better than 
## basic Decision Trees as the Data Set gets larger.

