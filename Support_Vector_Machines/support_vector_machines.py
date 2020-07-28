####################################################
###### Import Open-Software Python Libraries #######
####################################################

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

# Import load_breast_cancer function from datasets module of scikit-learn library.
from sklearn.datasets import load_breast_cancer

# Create an instance of the Breast Cancer Data Set that comes included with scikit-learn.
cancer_data = load_breast_cancer()

# Store data alone in its own DataFrame.
raw_data = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])

## Note: Every Data Set included in scikit-learn comes with a description field.

# Print the description field. Notice that :
    # 569 observations in the Data Set. 
    # 30 numeric attributes for each observation.
print(cancer_data['DESCR'])

#################################################################
###### Separate Data Set into Training Data and Test Data #######
#################################################################

# Specify x variables as raw_data pandas DataFrame.
x = raw_data

# Specify y variable as what's stored under the key target, parsed from the original cancer_data object.
y = cancer_data['target']


# Import function train_test_split from model_selection module of scikit-learn library.
from sklearn.model_selection import train_test_split

# Create Training Data and Test Data along both x and y axes.
    # Test Data is 30% of original Data Set.
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)

#######################################################
####### Train Our SVM Model and Make Predictions ######
#######################################################

# Import class SVC from svm module of scikit-learn library.
from sklearn.svm import SVC

# Create an object of class SVC.
model = SVC()

# Train the SVM Model by invoking method fit on object model.
    # Pass in x_training_data and y_training_data variables.
model.fit(x_training_data, y_training_data)

# Invoke method predict on the SVM Model.
    # Pass in array of values to generate predictions from.
predictions = model.predict(x_test_data)

###########################################################
######## Evaluate Decision Tree Model Performance #########
###########################################################

## Note: We are using Classification Report and Confusion Matrix to measure 
## performance for all our Machine Learning Classification Models.
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

# Generate a Classification Report
print(classification_report(y_test_data, predictions))
# Generate a Confusion Matrix
print(confusion_matrix(y_test_data, predictions))