# K-Nearest Neighbors

## Contents
* [Brief Description](#Brief-Description)
    * [Simple Example](#Simple-Example)
    * [Building a K-Nearest Neighbors Algorithm](#Building-a-K-Nearest-Neighbors-Algorithm)
    * [Importance of K](#Importance-of-K)
    * [Pros and Cons of K-Nearest Neighbors Algorithm](#Pros-and-Cons-of-K-Nearest-Neighbors-Algorithm)
* [Jupyter Notebook](#Jupyter-Notebook)
    * [Data Set](#Data-Set)
    * [Scaling](#Scaling)
    * [Picking Best K Value](#Picking-Best-K-Value)
        * [Elbow Method](#Elbow-Method)
* [Source](#Source)

## Brief Description
This is a Machine Learning Algorithm that can help you solve classification problems where there are more than 2 categories.
This is based on the simple principle that a data point belongs to the category with the majority of the K closest data points.

### Simple Example 
If categorizing football players and basketball players and (2/3) of the closest data points to the new data point are football players,
the K-Nearest Algorithm will predict that the new data point is also a football player.

### Building a K-Nearest Neighbors Algorithm
General Steps:
<ol>
    <li>Store the data.</li>
    <li>Calculate the <a href = "https://en.wikipedia.org/wiki/Euclidean_distance"><i>Euclidean distance</i></a> from the 
        new data point x to all the other data points.</li>
    <li>Sort the points in the Data Set in order of increasing distance from x.</li>
    <li>Predict using the same category as the majority of the K closest data points to x.</li>
</ol>

### Importance of K
Changing the value of K will change which category a new data point is assigned to:
<ul>
    <li>A low K value will cause your model to perfectly predict Training Data and poorly predict Test Data</li>
    <li>A too high K value will make your model unnecessarily complex.</li>
</ul>

The "best" value of K is where the error on the Test Data and Error on the Training Data are close to one another.
This results in overall low predictive error.

### Pros and Cons of K-Nearest Neighbors Algorithm
Main Advantages:
<ul>
    <li>The Algorithm is simple and easy to understand.</li>
    <li>It is trivial to train the model on new Training Data.</li>
    <li>It's easy to add more data to the Data Set.</li>
    <li>The model accepts only 2 parameters : K and the distance metric you'd like to use (e.g. <i>Euclidean distance</i>)</li>
</ul>

Main Disadvantages:
<ul>
    <li>There's a high computational cost to making predictions.</li>
    <li>It doesn't work well with categorical features.</li>
</ul>

## Jupyter Notebook
In our <a href= "https://nbviewer.jupyter.org/github/Dipto9999/ML-Introduction/blob/master/K-Nearest_Neighbors/k-nearest_neighbors.ipynb">Notebook</a>, we are working with an anonymous Data Set where the categories aren't known. This is very common in the real world (e.g. classified government data). 

### Data Set
Since we don't have much information regarding the Data Set, we must do as instructed and predict the 'TARGET CLASS' column. The Data Set is <a href = "https://nickmccullum.com/files/k-nearest-neighbors/classified_data.csv">downloaded</a> from Nick McCullum's <a href= "https://nickmccullum.com">Website</a>. 

### Scaling
It's necessary to scale the x-values of our Data Set to use the observations closest to the data point in order to make predictions. 
It is common practice to standardize the Data Set by adjusting every x-value so they're approximately on the same scale.

The <b>scikit-learn</b> library includes excellent functionality for this in the class <b>StandardScaler</b>.

``` python
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
```

### Picking Best K Value
Since the K value changes the predictive power and simplicity of our K-Nearest Neighbors Algorithm, we must be deliberate with the K value used to perform predictions on our Data Set.

#### Elbow Method
This involves iterating through a series of K values to pick the one with the lowest error rate when applied to our Test Data.

We can accomplish this by using a <b>Python</b> loop and appending error rates to a list as shown:

```python
    for i in np.arange (1, 101) :
        # For this test, we must create a new instance of class KNeighborsClassifier from scikit-learn.
            # Specify the parameter n_neighbors with the K value of i.
        new_model = KNeighborsClassifier(n_neighbors = i)
        # Train the model by fitting it to our Training Data.
        new_model.fit(x_training,data, y_training_data)
        # Make predictions on the Test Data.
        new_predictions = new_model.predict(x_test_data)
        # Calculate the mean difference for every incorrect prediction. 
        error_rates.append(np.mean(new_predictions != y_test_data))
```

<p align="center"><img src="Jupyter_Notebook-Preview.JPG" width="60%" height="60%" title="Preview of Notebook" ></p>

There is also a <a href = "k-nearest_neighbors.py">Python Script</a> written with additional comments to further understand the procedure of developing this Machine Learning Algorithm. 

## Source
<i>The information in this repository is derived from a FreeCodeCamp 
<a href= "https://www.freecodecamp.org/news/a-no-code-intro-to-the-9-most-important-machine-learning-algorithms-today">Article</a> written by Nick McCullum.</i>