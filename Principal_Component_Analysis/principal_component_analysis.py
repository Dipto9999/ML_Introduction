##########################################################
######### Import Open-Source Software Libraries ##########
##########################################################

# Include Libraries for Tabular Data
import pandas as pd
import numpy as np

# Include Libraries for Data Visualization
import matplotlib.pyplot as plt 
import seaborn as sns

# matplotlib visualizations will embed themselves 
# directly in our Jupyter Notebook. This will make them easier to 
# access and interpret.
%matplotlib inline

#######################################
######## Investigate Data Set #########
#######################################

# Load function load_breast_cancer from scikit-learn library.
from sklearn.datasets import load_breast_cancer

# Generate the Data Set by running this function.
raw_data = load_breast_cancer()

# Determine what type of data structure the object raw_data is.
    # Function will return sklearn.utils.Bunch, which is a special,
    # built-in data structure of scikit-learn. This behaves similar to 
    # a Python dictionary.
type(raw_data)

# One of the keys of this object is data, which can be used to
# transform the Data Set into a pandas DataFrame.   
raw_data_frame = pd.DataFrame(raw_data['data'], columns = raw_data['feature_names'])

# Can investigate what features the Data Set contains by printing array.
    # Will notice that it's a feature-rich Data Set.
print(raw_data_frame.columns)

## Side Note: The variable that we're trying to predict is held 
## within the object raw_data, under the key target. If a specific
## breast cancer tumor is: 
    ## malignant -> value is 1.
    ## benign -> value is 0.

## Note: It's difficult to perform exploratory data analysis on a Data Set using 
## traditional visualization techniques due to the many features. Perform 
## a principal component transformation to reduce the number of features to 2.

#####################################################################
############ Standardize the Data Set (Scale Features) ##############
#####################################################################

# Standardize the Data Set by transforming all the observations so that each feature is approximately the same size.

# Use class StandardScaler from scikit-learn library.
from sklearn.preprocessing import StandardScaler

# Instantitate an object of class StandardScaler.
data_scaler = StandardScaler()

# Train object data_scaler on raw_data_frame Data Set.
    # Lets object data_scaler observe the characteristics of each feature in 
    # Data Set prior to transforming each feature to the same scale.
data_scaler.fit(raw_data_frame)

# Create a new Data Set with standardized observations.
scaled_data_frame = data_scaler.transform(raw_data_frame)

######################################################################
############ Create Principal Component Analysis Model ###############
######################################################################

# Import class PCA from scikit-learn 
from sklearn.decomposition import PCA

# Instantiate an object of class PCA.
    # Specify parameter n_components to be 2.
model = PCA(n_components = 2)

# Fit object model on scaled_data_frame DataFrame
model.fit(scaled_data_frame)

# See the principal in action.
x_model = model.transform(scaled_data_frame)
    # Notice that transformed Data Set has 2 features.
print(x_model.shape)
    # Notice that original Data Set has 30 features.
    # It's nearly impossible to generate meaningful 
    # data visualizations from a Data Set of this size.
print(scaled_data_frame.shape)

## Note: Since principal components are linear combinations of the 
## original features in the Data Set, they are calculated by adding 
## and subtracting the original features of the Data Set.

# Generate coefficients of linear combination.
    # 2-D NumPy array generated with 2 rows (i.e. principal components) and 30 columns (i.e. original features).
    # Values of each item in array correspond to coefficient on that specific feature in the Data Set.
        # Equation for first principal component is in the form 0.21890244x1 + 0.10372458x2 + ...
model.components_

#################################################
########### Data Visualization ##################
#################################################

# Create a simple scatterplot from the 2 principal components.
    # Generated visualization shows each data point 
    # as a function of the principal components.
plt.scatter(x_model[:, 0], x_model[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
    # Modify the scatterplot to add a color scheme.
        # Each data point is colored depending on whether
        # it represents a benign or malignant tumor.
plt.scatter(x_model[:, 0], x_model[:, 1], c = raw_data['target'])

