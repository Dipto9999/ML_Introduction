# Import Library for Working with Tabular Data
import pandas as pd
# Import Library for Numerical Computing
import numpy as np
# Import Library for Data Visualization
import matplotlib.pyplot as plt 
# Import Another Library for Data Visualizations
# This makes it easier to create beautiful data visualizations using matplotlib.
import seaborn as sns

# matplotlib visualizations will embed themselves 
# directly in our Jupyter Notebook. This will make them easier to 
# access and interpret.
%matplotlib inline 

# Import Housing Data Set into Jupyter Notebook Under raw_data Variable
raw_data = pd.read_csv('Housing_Data.csv')

# Can use info method to get some high-level information about the data set.
raw_data.info()

# Can also use seaborn method pairplot to learn about this data set.
# This passes in the entire DataFrame as a parameter and provides a visual
# model of the data set as opposed to above.
sns.pairplot(raw_data)

# Can generate a list of the DataFrame's columns
# Will use all of these variables in x-array except for:
    # Price (which we are trying to predict)
    # Address (contains text)
raw_data.columns

# Create our x-array and assign it to a variable called x
x = raw_data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
    'Avg. Area Number of Bedrooms', 'Area Population']]

# Similarly, create our y-array and assign it to a variable y
y = raw_data['Price']

# Split our data set into Training Data and Test Data.
    # Must import function train_test_split from model_selection
    # module of scikit-learn.
from sklearn.model_selection import train_test_split

# We want the Test Data to be 30% of the Entire Data Set.
    # train_test_split function returns a Python list of length 4,
    # where the items are x_train ... y_test respectively.
    # List unpacking is used to assign the proper values to the correct variable names.
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)

# Must import the LinearRegression estimator from scikit-learn in order to
# build and train the linear regression machine learning model.
from sklearn.linear_model import LinearRegression

# Create an instance of the Linear Regression Python object and
# assign it to a variable called model.
model = LinearRegression()

# Can use scikit-learn's fit method to train this model on our Training Data.
model.fit(x_train, y_train)

# At this point model has been trained.
    # Can now examine each of the model's coefficients by printing them in this format.
print(model.coef_)
    # Can also view the coefficients by placing them in a DataFrame.
    # This organizes the output with labels in a tablelike format.
pd.DataFrame(model.coef_, x.columns, columns = ['Coeff'])

## Side Note: Coefficients quantify the impact of the value of the specified
## variable on the predicted variable. It makes the assumption that all other variables
## are held constant. (i.e. Coefficient 15 for Area Population means that a 1-unit increase in
## the variable will result in a 15-unit increase in the predicted variable Price.)

# Can similarly see the intercept of the regression equation.
print(model.intercept_)

# Call the predict method on the model variable to make predictions from a machine learning
# model using scikit-learn. 
    # The predictions variable holds the predicted values of the features stored
    # in x_test.
predictions = model.predict(x_test)

# Plot the real values in y_test array against the predictions array using a matplotlib scatterplot.
    # A perfectly straight diagonal line in our scatterplot would indicate that our model perfectly predicted the 
    # y_array values.
plt.scatter(y_test, predictions)

# Can also plot residuals to visually assess the performance of our model. 
# These are the difference between the actual y_array and the predicted y_array values.
    # If the residuals from our machine learning model appear to be normally distributed,
    # this is a good sign that we have selected an appropriate model type (i.e. linear regression)
    # to make predictions from our data set. 
plt.hist(y_test - predictions)

# Three main performance metrics used for regression machine learning models:
    # Mean Absolute Error
    # Mean Squared Error
    # Root Mean Squared Error

# Must run following import statement in Jupyter Notebook.
from sklearn import metrics

# Calculate Mean Absolute Error in Python
metrics.mean_absolute_error(y_test, predictions)
# Calculate Mean Squared Error in Python
metrics.mean_squared_error(y_test, predictions)
# Calculate Root Mean Squared Error (scikit-learn doesn't have built in method.)
np.sqrt(metrics.mean_squared_error(y_test, predictions))





