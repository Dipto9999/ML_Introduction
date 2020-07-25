# Linear Regression

## Contents
* [Brief Description](#Brief-Description)
* [Jupyter Notebook](#Jupyter-Notebook)
    * [Data Set](#Data-Set)
    * [Coefficients](#Coefficients)
* [Source](#Source)

## Brief Description
Linear Regression is a Machine Learning Algorithm which is used to predict some y values based on a
set of x values. The end product of a regression model is an equation that you can use to predict the y-value of an x-value.
Note that you don't know the y-value in advance. 

## Jupyter Notebook
In our <a href= "https://nbviewer.jupyter.org/github/Dipto9999/ML-Introduction/blob/master/Linear_Regression/linear_regression.ipynb">Notebook</a>, we are working with a Data Set of housing data and attempt to predict housing prices.

### Data Set
The Data Set contains a number of features which are randomly generated. It is <a href = "https://nickmccullum.com/files/Housing_Data.csv">downloaded</a> from Nick McCullum's <a href= "https://nickmccullum.com">Website</a>.
These include:
<ul>
    <li>Average Income in the Area of the House</li>
    <li>Average Number of Total Rooms in the Area</li>
    <li>Price the House is Sold For</li>
    <li>Address of the House</li>
</ul>

### Coefficients
These quantify the impact of the value of the specified variable on the predicted variable. It makes the assumption that 
all other variables are held constant. (i.e. Coefficient 15 for Area Population means that a 1-unit increase in the variable 
will result in a 15-unit increase in the predicted variable Price.)

<p align="center"><img src="Jupyter_Notebook-Preview.JPG" width="60%" height="60%" title="Preview of Notebook" ></p>

There is also a <a href = "linear_regression.py">Python Script</a> written with additional comments to further understand the procedure of developing this Prediction Algorithm. 

## Source
<i>The information in this repository is derived from a FreeCodeCamp 
<a href= "https://www.freecodecamp.org/news/a-no-code-intro-to-the-9-most-important-machine-learning-algorithms-today">Article</a> written by Nick McCullum.</i>