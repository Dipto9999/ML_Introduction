#######################################
######## Investigate Data Set #########
#######################################

# Import function make_blobs to create artificial data.
from sklearn.datasets import make_blobs

# Create a Data Set with 200 samples that has 2 features and 4 cluster centers.
    # Set the standard deviation within each cluster to 1.8.
raw_data = make_blobs(n_samples = 200, n_features = 2, centers = 4, cluster_std = 1.8)

# Notice that object raw_data is a Python tuple. 
    # The first element of this tuple is a NumPy array with 200 observations, 
    # # each containing 2 features as specified in our function make_blobs. 
print(raw_data)

####################################################
###### Import Open-Software Python Libraries #######
####################################################

# Import Libraries for Manipulating Large Data Sets
import pandas as pd
import numpy as np

# Import Libraries for Creating Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# matplotlib visualizations will embed themselves 
# directly in our Jupyter Notebook. This will make them easier to 
# access and interpret.
%matplotlib inline

#################################################
########## Creating Data Visualizations #########
#################################################

# Verify our Data Set has 4 cluster centers as specified in function make_blobs.
    # Plot all rows in the first column of Data Set against all rows in second coumn of Data Set.
plt.scatter(raw_data[0][:, 0], raw_data[0][:, 1])

# Modify the scatterplot to color code each data cluster.
    # Need to reference the second element of object raw_data (i.e. NumPy array that contains the cluster to which each observation belongs).
plt.scatter(raw_data[0][:, 0], raw_data[0][:, 1], c = raw_data[1])

####################################################
###### Train our K-Means Clustering Algorithm ######
####################################################

# Import K-Means Clustering Algorithm from scikit-learn library.
from sklearn.cluster import KMeans

# Create instance of class KMeans with parameter n_clusters = 4.
model = KMeans(n_clusters = 4)

# Invoke method fit on object model by passing in first element of object raw_data.
model.fit(raw_data[0])

## Note: When building Unsupervised Machine Learning Models, 
## a split isn't made to separate Training Data and Test Data.

############################################################
###### Make Predictions With K-Means Clustering Model ######
############################################################

# Predict which cluster each data point belongs to.
    # Access labels_ attribute from the object model to
    # generate a NumPy array with predictions for each data point.
model.labels_

# See where the center of each cluster lies.
    # Access cluster_centers_ attribute from the object model to generate
    # a two-dimensional NumPy array containing the coordinates of each cluster's center.
model.cluster_centers_

#########################################################################
######## Visually Evaluate Accuracy of K-Means Clustering Model #########
#########################################################################

# Visualize the accuracy of the K-Means Clustering Model 
# by generating two different plots side-by-side.
    # Note that the coloring between the two plots may be different.
f, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (10, 6))
        # One plot shows the clusters according to our Machine Learning Model.
            # Will notice that the model wasn't perfect. 
            # Data points are occasionally misclassified, usually along a cluster's edge.
ax1.set_title('Our Model')
ax1.scatter(raw_data[0][:, 0], raw_data[0][:, 1], c = model.labels_)
        # One plot shows the clusters according to real Data Set.
ax2.set_title('Original Data')
ax2.scatter(raw_data[0][:, 0], raw_data[0][:, 1], c = raw_data[1])

## Note: When measuring the predictive accuracy of a K-Means Clustering Model,
## practitioners often don't know the clusters in advance. This is due to the 
## fact that the K-Means Clustering Machine Learning Algorithm is used to find 
## patterns that aren't obvious in a Data Set.
