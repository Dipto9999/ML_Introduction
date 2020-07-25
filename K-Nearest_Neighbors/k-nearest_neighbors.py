#########################################################################################################
########## Import Open-Source Python Libraries (Similar to Previous Machine Learning Algorithms) ########
#########################################################################################################

# Import Library for Numerical Computing
import numpy as np
# Import Library for Working with Tabular Data
import pandas as pd
# Import Library for Data Visualization
import matplotlib.pyplot as plt
# Import Another Library for Data Visualizations
# This makes it easier to create beautiful data visualizations using matplotlib.
import seaborn as sns

# matplotlib visualizations will embed themselves 
# directly in our Jupyter Notebook. This will make them easier to 
# access and interpret.
%matplotlib inline

#####################################################
### Import Classified Data Set into Python Script ###
#####################################################

# Import file which contains Data Set into Python script using pandas library.
    # The parameter index_col is specified to remove an unnamed column whose values are equal to the DataFrame's index.
raw_data = pd.read_csv('classified_data.csv', index_col = 0)
# Print the DataFrame inside Jupyter Notebook to get a sense of what the Data Set looks like.
raw_data
# Print a list of the Data Set's column names.
    # Meaning is ambiguous since the Data Set is classified.
    # Will notice that every column is numerical in nature, which makes modelling it easier with Machine Learning Algorithm.
print(raw_data.columns)

###########################################
########## Scaling the Data Set ###########
###########################################

## Note: The scale of the features within a Data Set matters since you
## need to use the observations closest to the data point in order to make predictions.
## For this reason, Machine Learning Practitioners standardize the Data Set a lot of the time
## by adjusting every x-value so they're approximately on the same scale.

# Import the class StandardScaler from scikit-learn.
    # This behaves similar to the class LinearRegression and LogisticRegression. 
from sklearn.preprocessing import StandardScaler

# Instantiate the class StandardScaler to fit the instance on Data Set.
scaler = StandardScaler()

# Train the object scaler on Data Set.
scaler.fit(raw_data.drop('TARGET CLASS', axis = 1))

# Standardize all of the features in the Data Set to be approximately the same scale.
    # The method transform creates a NumPy array of the features. 
scaled_features = scaler.transform(raw_data.drop('TARGET CLASS', axis = 1))

# Create a pandas DataFrame of the features.
scaled_data = pd.DataFrame(scaled_features, columns = raw_data.drop('TARGET CLASS', axis = 1).columns)

###############################################################
###### Split Data Set into Training Data and Test Data ########
###############################################################

# Import method train_test_split from module model_selection of scikit-learn.
from sklearn.model_selection import train_test_split

# Specify x-values as the scaled_data DataFrame we created.
x = scaled_data
# Specify y-values as the TARGET CLASS column of raw_data DataFrame.
y = raw_data['TARGET CLASS']

# Test Data should have a size of 30% the Data Set.
    # Use list unpacking to create Training Data and Test Data.
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)

###############################################################
########### Training Our K-Nearest Neighbors Model ############
###############################################################

# Import the appropriate model from scikit-learn (i.e. K-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier

# Instantiate the class KNeighborsClassifier
    # Specify the parameter n_neighbors with the K value of 1.
model_k_1 = KNeighborsClassifier(n_neighbors = 1)

# Fit the K-Nearest Neighbors model to our Training Data
model_k_1.fit(x_training_data, y_training_data)

#########################################################################################
################### Make Predictions and Measure Performance (K Value of 1) #############
#########################################################################################

predictions_k_1 = model_k_1.predict(x_test_data)

# Import functions from scikit-learn library to measure the performance of our model.
    # Library contains built-in functions for measuring performance of Machine Learning Classification Models.
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Generate a Classification Report
print(classification_report(y_test_data, predictions_k_1))

# Generate a Confusion Matrix
print(confusion_matrix(y_test_data, predictions_k_1))

###########################################################################
########### Choosing an Optimal K Value using the Elbow Method ############
###########################################################################

# Loop through K values from 1 to 100 and select the value with the lowest error rate when applied to our Test Data.
# This is done by appending the error rates for the K values to a list error_rates.
    # Create an empty list
    error_rates = []

    for i in np.arange (1, 101) :
        # For this test, we must create a new instance of class KNeighborsClassifier from scikit-learn.
            # Specify the parameter n_neighbors with the K value of i.
        new_model = KNeighborsClassifier(n_neighbors = i)
        # Train the model by fitting it to our Training Data.
        new_model.fit(x_training_data, y_training_data)
        # Make predictions on the Test Data.
        new_predictions = new_model.predict(x_test_data)
        # Calculate the mean difference for every incorrect prediction. 
        error_rates.append(np.mean(new_predictions != y_test_data))

# Visualize how our error rate changes with different K values using a matplotlib visualization.
    # Will notice that the error rates tend to be minimized with K value of approximately 50. This
    # would balance both simplicity and predictive power.
plt.plot(error_rates)

#####################################################################################################################################
################### Train Our K-Nearest Algorithm Model, Make Predictions, and Measure Performance (K Value of 50) ##################
#####################################################################################################################################

# Instantiate the class KNeighborsClassifier
    # Specify the parameter n_neighbors with the K value of 50.
model_k_50 = KNeighborsClassifier(n_neighbors = 50)

# Fit the K-Nearest Neighbors model to our Training Data
model_k_50.fit(x_training_data, y_training_data)

# Store the predictions of Test Data.
predictions_k_50 = model_k_50.predict(x_test_data)

# Generate a Classification Report
print(classification_report(y_test_data, predictions_k_50))

# Generate a Confusion Matrix
print(confusion_matrix(y_test_data, predictions_k_50))