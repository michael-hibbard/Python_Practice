# Created by Michael Hibbard 10/21/16 for practice with Scikit-learn linear regression

# Import necessary libraries
import random as random
import math as math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# # Step 1: Fit something that we know the answer to:
#
# # Initialize a list of data values
# dataValues = []
#
# # Populate the list
# for i in range (0,100):
#     xValue = i
#     yValue = 2*xValue + 4
#     dataValues.append([xValue,yValue])
#
# # Determine how much data to have in testing set and how much data to have in training set
# numberToTest = int(input("There are 100 data points. How many would you like to add to the testing set?"))
#
# # Randomly select the data to be testing data and training data
# # Initialize a testing data array
# testingData = []
# for i in range (0,numberToTest):
#     # Randomly choose an index to remove
#     index = int(random.random() * dataValues.__len__())
#     # Store the data to add to testing set
#     dataToRemove = dataValues[index]
#     # Remove the data from training set
#     dataValues.pop(index)
#     # Add the data to the testing set
#     testingData.append(dataToRemove)
#
# # For aesthetic purposes, sort the testing data set
# testingData.sort()
#
# # Break up the training data into x and y
# xTrainingData = []
# yTrainingData = []
# for i in range (0,dataValues.__len__()):
#     xTrainingData.append(dataValues[i][0])
#     yTrainingData.append(dataValues[i][1])
#
# # Format the data
# xTrain = np.array(xTrainingData)
# xTrain = xTrain.reshape((-1,1))
# yTrain = np.array(yTrainingData)
# yTrain = yTrain.reshape((-1,1))
#
# # Create a regression model to use on the training data
# regr = linear_model.LinearRegression()
#
# # fit the data to the linear regression model
# regr.fit(xTrain,yTrain)
#
# # Print out the results of the linear regression training:
# print("the linear formula for the regression is: ",regr.coef_[0][0],"x + ",regr.intercept_[0]," .")

# # # # # # # # # #

# Step 2: Fit something we know the answer to, but add some variability

# Initialize the data values
dataValues = []

# Populate the data values
for i in range(0,1000):
    xValue = i
    # Define an error term between -1 and 1 and add it to the dataValues list
    delta = -5 + random.random() * 10
    yValue = 2*xValue + 4 + delta
    dataValues.append([xValue,yValue])

# Do some preliminary analysis of the data:
bins = []
for i in range(0,100):
    bins.append(10*i)
yHist = plt.figure(0)
yValues = []
for i in range(0,dataValues.__len__()):
    yValues.append(dataValues[i][1])
yValues = np.array(yValues)
yValues = yValues.reshape((-1,1))
plt.hist(yValues,bins,histtype='bar')
yHist.suptitle('Histogram of y data values')
print("The mean of the data was",yValues.mean()," and the standard deviation was",yValues.std())


# Define the number of data to test each run:
numberToTest = .2*dataValues.__len__()

# initialize a table of RMSE values and set a placeholder for the highest value of the RMSE
RMSEvalues = []
bestRMSE = 100
bestActual = None
bestPredicted = None

# Want to repeat the following process 100 times. Use a for loop.
for k in range(0,100):

    # Randomly select the data, also need to declare the data values as a new variable so that it will be "refilled"
    # with each run.
    testingData = []
    dataValuesPlaceholder = dataValues.copy()
    for i in range(0,int(numberToTest)):
        # Randomly choose an index to remove
        index = int(random.random() * dataValuesPlaceholder.__len__())
        # Store the data to add to testing set
        dataToRemove = dataValuesPlaceholder[index]
        # Remove the data from training set
        dataValuesPlaceholder.pop(index)
        # Add the data to the testing set
        testingData.append(dataToRemove)
    testingData.sort()

    # Break up the training data into x and y
    xTrainingData = []
    yTrainingData = []
    for i in range (0,dataValuesPlaceholder.__len__()):
        xTrainingData.append(dataValuesPlaceholder[i][0])
        yTrainingData.append(dataValuesPlaceholder[i][1])

    # Format the data
    xTrain = np.array(xTrainingData)
    xTrain = xTrain.reshape((-1,1))
    yTrain = np.array(yTrainingData)
    yTrain = yTrain.reshape((-1,1))

    # # Graph a scatter plot of the training data
    # scatter = plt.figure(0)
    # plt.scatter(xTrain,yTrain,marker="o",color="b")
    # scatter.suptitle('Scatter plot of training data')

    # # Graph a histogram of the training data
    # bins = []
    # for i in range(0,220):
    #     if i % 10 is 0:
    #         bins.append(i)
    # hist = plt.figure(1)
    # plt.hist(yTrain,bins,histtype='bar')
    # hist.suptitle('Histogram of Training Data')

    # Create a regression model to use on the training data
    regr = linear_model.LinearRegression()

    # fit the data to the linear regression model
    regr.fit(xTrain,yTrain)

    # # Print out the results of the linear regression training:
    # print("the linear formula for the regression is: ",regr.coef_[0][0],"x + ",regr.intercept_[0],". Furthermore, for the",
    #             " training data, the mean of the data is",yTrain.mean(), "and the standard deviation is",yTrain.std(),".")

    # Use the results of the training to fit the testing data:
    a = regr.coef_
    b = regr.intercept_

    # Create separate testing data arrays:
    xTestingData = []
    yTestingData = []
    for i in range(0,testingData.__len__()):
        xTestingData.append(testingData[i][0])
        yTestingData.append(testingData[i][1])

    # Using the x testing data, fit new values
    yTestPredicted = []
    for i in range(0,xTestingData.__len__()):
        yTestPredicted.append(a*xTestingData[i]+b)

    # Format the data
    xTest = np.array(xTestingData)
    xTest = xTest.reshape((-1,1))
    yTestActual = np.array(yTestingData)
    yTestActual = yTestActual.reshape((-1,1))
    yTestPredicted = np.array(yTestPredicted)
    yTestPredicted = yTestPredicted.reshape((-1,1))

    # # Plot the predicted vs actual
    # scatterPredicted = plt.figure(2)
    # plt.scatter(yTestActual,yTestPredicted,marker="o",color="b")
    # scatterPredicted.suptitle('Predicted vs. actual Data')
    # plt.xlabel('Actual Values')
    # plt.ylabel('Predicted Values')

    # Add the RMSE of the data to the RMSE list:
    RMSE = mean_squared_error(yTestActual,yTestPredicted)
    RMSEvalues.append(RMSE)

    # Check whether or not to update the value of the highest RMSE
    if RMSE < bestRMSE:
        bestRMSE = RMSE
        bestActual = yTestActual
        bestPredicted = yTestPredicted

print("The lowest RMSE of the CV tests was",bestRMSE)
RMSEvalues = np.array(RMSEvalues)
RMSEvalues = RMSEvalues.reshape((-1,1))
print("The mean RMSE was",RMSEvalues.mean()," and the standard deviation was",RMSEvalues.std())

# Graph a histogram of the RMSE values
bins = []
for i in range(0,35):
    bins.append(.1*i+6.5)
hist = plt.figure(1)
plt.hist(RMSEvalues,bins,histtype='bar')
hist.suptitle('Histogram of RMSE values')

# Graph a scatterplot of the best data:

if bestActual is not None:
    scatter = plt.figure(2)
    plt.scatter(bestActual,bestPredicted,c='y')
    scatter.suptitle('Actual vs. Predicted Values for Lowest RMSE')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

plt.show()