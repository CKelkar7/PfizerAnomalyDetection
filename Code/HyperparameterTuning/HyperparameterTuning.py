#!/usr/bin/env python
# coding: utf-8

# ### Preliminary Code

#Importing the necessary packages
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Flatten
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
import matplotlib.pyplot as pltport


#Reading in the Pfizer stock price data
stock_prices = pd.read_csv('C:/Users/chink/Downloads/PFE.csv')
stock_prices


# This dataset includes dates ranging from June 1, 1972 to October 1st, 2021. It has 5 different stock prices for each date: "Open", "High", "Low", "Close", and "Adj Close". I will choose to use the adjusted closing price as this quantity accounts for any corporate actions that affect the closing price of the stock, such as stock splits or rights offerings.

#Visualizing prices
fig = px.line(stock_prices, x='Date', y='Adj Close',)
fig.update_layout(
    title="Pfizer Stock Price",
    xaxis_title="Date",
    yaxis_title="Adjusted Closing Price ($)")
fig.show()


#Checking for null entries in the "Date" column
if np.sum(stock_prices['Date'].isnull()) == 0:
    print("There are no null entries in the Date column")
else:
    print("There are null entries in the Date column")


#Checking for null entries in the "Adj Close" column
if np.sum(stock_prices['Date'].isnull()) == 0:
    print("There are no null entries in the Adj Close column")
else:
    print("There are null entries in the Adj Close column")



# ### Splitting and scaling the data

# To split the data, I will select a training and a test set. The test set is of course stock prices from the beginning January 2020 to the end September 2021. For the training set, I will select a subset of the data during which time I assume that there are almost zero anomalies. While there are many ways to construct this dataset, I will choose the decade of the 2010s as the training data, as this constitutes a large chunk of time and immediately precedes the test data.


#Creating the training set
start_Jan_2010_ind = stock_prices[stock_prices['Date'].str.contains('2010-01')].index[0]
end_Dec_2019_ind = stock_prices[stock_prices['Date'].str.contains('2019-12')].index[-1]
training_data = stock_prices.loc[start_Jan_2010_ind:end_Dec_2019_ind][['Date','Adj Close']]


# Just a small note: due to weekends and holidays, the beginning date for Jan 2010 isn't just '2010-01-01' and likewise the end date of 2019 isn't just '2019-12-31'; for this reason, I need to collect the indices corresponding to the start and end dates
# for the period of time spanning the training set

#Viewing the training set
training_data


#Creating the test set in a similar fashion
start_Jan_2020_ind = stock_prices[stock_prices['Date'].str.contains('2020-01')].index[0]
end_Sep_2021_ind = stock_prices[stock_prices['Date'].str.contains('2021-09')].index[-1]
test_data = stock_prices.loc[start_Jan_2020_ind:end_Sep_2021_ind][['Date','Adj Close']]


#Viewing the test data
test_data


#Scaling the data
scale = MinMaxScaler()
scale.fit(training_data[['Adj Close']])
train = training_data.copy()
test = test_data.copy()
train['Adj Close'] = scale.transform(training_data[['Adj Close']])
test['Adj Close'] = scale.transform(test_data[['Adj Close']])

# ### Creating the time-series sequences


#Creating a function to construct the sequences
def sequences(df,t_step):
    input_seq = []
    output_val = []
    for i in range(len(df)-t_step):
        sequence = df.iloc[i:i+t_step]['Adj Close'].values
        input_seq.append(sequence)
        val = df.iloc[i+t_step]['Adj Close']
        output_val.append(val)
    #input_seq = np.array(input_seq)
    newshape = (len(input_seq),30,1)
    input_seq = np.reshape(input_seq,newshape)
    input_seq = np.array(input_seq)
    output_val = np.array(output_val)
    return input_seq,output_val

#Creating the train sequences, predicted train stock prices, test sequences, and predicted test sequences
time_step = 60
train_sequences, train_sp = sequences(train,time_step)
test_sequences, test_sp = sequences(test,time_step)



#Confirming the shapes of the datasets are correct
print("Training sequence data shape: ",train_sequences.shape)
print('Training stock price data shape: ',train_sp.shape)
print('Testing sequence data shape: ',test_sequences.shape)
print('Testing stock price data shape: ',test_sp.shape)


#Setting a seed for reproducibility
np.random.seed(2021)
tf.random.set_seed(2021)

#Defining the model
def create_model(dropout_rate=0.15):
    model = Sequential()
    model.add(LSTM(100,activation = 'tanh', input_shape=(train_sequences.shape[1],train_sequences.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(RepeatVector(train_sequences.shape[1]))
    model.add(LSTM(100, activation = 'tanh', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(TimeDistributed(Dense(train_sequences.shape[2])))
    model.compile(optimizer='adam', loss='mae')
    return model

#Creating the model
model = KerasRegressor(build_fn=create_model,verbose=0)

#Defining the grid-search parameters
dropout_rate = [0.15,0.2,0.25,0.3]
batch_size_list=[25,50,75,100]
epochs_list=[100,150,200,250]

#Define the parameter grid
param_grid = dict(dropout_rate=dropout_rate,batch_size=batch_size_list,epochs=epochs_list)

#Grid-searching and tuning the hyperparamters
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(train_sequences,train_sequences)

#Outputting the results to a file
file = open('C:/Users/chink/Downloads/GridSearchOutput.txt','w')
print(grid_result.best_score_,grid_result.best_params_)
a = "Best score: "+str(grid_result.best_score_,)+"\n"+"Best_params: "+str(grid_result.best_params_)
file.write(a)
file.close()






