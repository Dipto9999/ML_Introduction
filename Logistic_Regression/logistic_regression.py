######################################################################################################
### Import Open-Source Software Libraries as with Linear Regression Model Under Conventional Alias ###
######################################################################################################

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

#########################################################################################
#################### Import Titanic Data Set into Python Script #########################
#########################################################################################

titanic_data = pd.read_csv('titanic_train.csv')

# Use titanic_data DataFrame to investigate the data included in the Titanic Data Set.

# This method prints the specified number of rows of the DataFrame (i.e. 5).
titanic_data.head(5)
# This method shows the column names in the DataFrame.
titanic_data.columns

## Side Note: For the "Survived" data point, the variable will
## hold a value of 1 if the passenger survived and 0 if the
## passenger didn't survive.

#########################################################################################
###### Exploratory Data Analysis Prior to Training Logistic Regression Model ############
#########################################################################################

# It's useful to have a sense of the ratio between classification categories 
# (i.e. how many survivors vs non-survivors in our training data). Let's visualize this.
sns.countplot(x = 'Survived', data = titanic_data)

# Compare survival rates relative to some data feature like Male and Female values for Sex Variable.
    # Will notice that passengers with sex Male were more likely to be non-survivors than passengers with sex Female.
sns.countplot(x = 'Survived', hue = 'Sex', data = titanic_data)

# Similarly, compare survival rates relative to PClass Variable.
    # Will notice that passengers with PClass value of 3 (cheapest and least luxurious class) were most likely to die. 
sns.countplot(x = 'Survived', hue = 'Pclass', data = titanic_data)

# Generate a histogram to see Age distribution of passengers.
# Use dropna() method to account for data set containing several null values.
    # Will notice that there is a concentration of passengers with Age value between 20 and 40.
plt.hist(titanic_data['Age'].dropna())

# Investigate the distribution of Fare Prices within Titanic Data Set.
    # Will notice that there are three distinct groups of Fare Prices within the data set.
    # This probably corresponds to the different PClass categories.
plt.hist(titanic_data['Fare'])

# Examine where the Titanic Data Set contains missing data. 
    # The function isnull() will generate a DataFrame of boolean values where cell contains: 
        # True if null value
        # False otherwise
titanic_data.isnull()

    # A quicker visualization for assessing missing data is by 
    # using seaborn visualization library to create a heatmap.
        # White lines indicate missing values in Data Set 
        # (Will notice that the majority is in the Age and Cabin columns)
sns.heatmap(titanic_data.isnull(), cbar = False)

#######################################################################################################################
############ Fill in Missing Data Using Some Form of Mathematics (Only if Small Amount is Missing in Column) ##########
#######################################################################################################################

# Imputation Method fills in missing Age values with average Age value 
# for the specific Pclass the passenger belongs to.
    # Generate a boxplot of Age distributions in each Pclass.
        # Will notice that passengers with Pclass value of 1 tend to be the oldest.
        # Similarly, passengers with Pclass value of 3 tend to be the youngest.
        # It is also assumed this probably correlates with Ticket Price as well.
sns.boxplot(titanic_data['Pclass'], titanic_data['Age'])

    # Build a custom function to perform Imputation.
def impute_missing_age(columns) :
    age = columns[0]
    passenger_class = columns[1]

    # Check if Age value is missing.
    if pd.isnull(age) :
        if (passenger_class == 1) :
            # Return average value of Pclass 1.
            return titanic_data[titanic_data['Pclass'] == 1]['Age'].mean()
        elif (passenger_class == 2) :
            # Return average value of Pclass 2.
            return titanic_data[titanic_data['Pclass'] == 2]['Age'].mean()
        elif (passenger_class == 3) :
            # Return average value of Pclass 3.
            return titanic_data[titanic_data['Pclass'] == 3]['Age'].mean()
    else :
        return age

    # Apply custom function to every row in titanic_data DataFrame
titanic_data['Age'] = titanic_data[['Age','Pclass']].apply(impute_missing_age, axis = 1)

    # Check original heatmap to notice that Age column is no longer missing data.
sns.heatmap(titanic_data.isnull(), cbar = False)

## Note: It was rather important we dealt with the missing Age values 
## since this datapoint has an impact on survival for most disasters and diseases.

#################################################################################################################
######## Remove Column with High Prevalence of Missing Data (Unwise to Perform Imputation in this Case) #########
#################################################################################################################

titanic_data.drop('Cabin', axis = 1, inplace = True)

# Similarly, remove any additional columns that contain missing data
titanic_data.dropna(inplace = True)

# Check original heatmap to notice that there isn't missing data in any column.
sns.heatmap(titanic_data.isnull(), cbar = False)

#########################################################################################
############## Must Numerically Work with Categorical Features ##########################
#########################################################################################

# Create dummy variables to solve this issue by creating a 
# new column for each value in DataFrame column.
    # Output will create two new columns, Male and Female.
    # These are perfect predictors of each other and significantly 
    # reduce the predictive power of our algorithm (i.e. Multicollinearity). 
    # (e.g. 0 in Female column indicates a 1 in Male column)
pd.get_dummies(titanic_data['Sex'])
    # Add argument drop_first to method get_dummies to remove Multicollinearity from our model.
sex_data = pd.get_dummies(titanic_data['Sex'], drop_first = True)
    # For our embarked_data variable, we have 2 columns (i.e. Q and S).
        # Have removed C column. Note that Q and S columns are not perfect predictors of each other. 
embarked_data = pd.get_dummies(titanic_data['Embarked'], drop_first = True)

# Concatenate sex_data and embarked_data data columns into existing pandas DataFrame.
titanic_data = pd.concat([titanic_data, sex_data, embarked_data], axis = 1)
    # Check if concatenation done successfully.
print(titanic_data.columns)

# Drop original Sex and Embarked columns from DataFrame for better readability.
# Also, drop columns that aren't predictive of Titanic crash survival rates for same reason (i.e. Name, PassengerId, Ticket).
titanic_data.drop(['Name', 'Ticket', 'Sex', 'Embarked'], axis = 1, inplace = True)
    # Check if columns are dropped successfully.
print(titanic_data.columns)

# At this point, every field in the Data Set is numeric, making it an excellent 
# candidate for a Logistic Regression Machine Learning Algorithm.

########################################################################################################################
########### Split our titanic_data DataFrame into Training Data and Test Data (Similar to Linear Regression) ###########
########################################################################################################################

# y-values -> data we're trying to predict.
y_data = titanic_data['Survived']
# x-values -> data used to make predictions.
x_data = titanic_data.drop('Survived', axis = 1)

# Import function train_test_split  from scikit-learn
from sklearn.model_selection import train_test_split

# Combine function train_test_split with list unpacking.
    # Test Data is 30% of original Data Set.
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.3)

#########################################################################################
######################## Train Our Logistic Regression Model ############################
#########################################################################################

# Import the appropriate model from scikit-learn (i.e. Logistic Regression)
from sklearn.linear_model import LogisticRegression
# Instantiate the LogisticRegression object to create our model.
    # Increase max_iter variabe from default value of 100. Otherwise,
    # Jupyter Notebook gives ConvergenceWarning.
model = LogisticRegression(max_iter = 1000)

# Call method fit on LogisticRegression object by passing in our x_training_data and y_training_data variables.
model.fit(x_training_data, y_training_data)

#########################################################################################
################### Make Predictions and Measure Performance ############################
#########################################################################################

predictions = model.predict(x_test_data)

# classification_report is a built in module in scikit-learn 
# which makes measuring the performance and accuracy of a Classification 
# Machine Learning Model easier.
from sklearn.metrics import classification_report

# Calculate the performance metrics for our Logistic Regression Machine Learning module.
classification_report(y_test_data, predictions)

# Can see the raw Confusion Matrix and calculate the performance metrics manually as well.
    # This is a tool used to compare True Positives, True Negatives, False Positives, False Negatives.
    # Allows you to assess whether your model is particularly weak in a specific quadrant.
        # You may wish to ensure that the model performs especially well in a dangerous zone of the Confusion Matrix.
        # (e.g. High rate of False Negatives for cancer diagnosis 
        # indicates that you incorrectly predict malignant tumors to be non-malignant.)
from sklearn.metrics import confusion_matrix
print (confusion_matrix(y_test_data, predictions))

