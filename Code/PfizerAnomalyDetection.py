#!/usr/bin/env python
# coding: utf-8

# # Identifying Anamolies in Pfizer Stock Data Using an LSTM Autoencoder

# In this notebook, I will use an LSTM autoencoder to identify anomalies in the Pfizer stock price from January 2020 through September 2021. 
#     
# Coming from a biological background, I am naturally drawn to the analysis of pharmaceutical stock data. Additionally, as the word is currently coming out of the tail end of a pandemic, the Pfizer vaccine has made the company one of the most consequential stocks out there.
#     
# The approach is inspired by that of TareqTayeh (1), whose Github is linked at the bottom of this document; the Pfizer stock data was obtained from Kaggle (2). I used MachineLearningMastery’s tutorials (3,4) on LSTM autoencoders and hyperparameter tuning. Lastly, I used Yahoo Finance to check my predicted anomalies against Pfizer’s stock prices (5). 
#     
# The approach consists of six main steps:
#     
#     1) Split and scale the data
#     2) Create the sequences
#     3) Build the LSTM autoencoder
#     4) Train model and run on the test data
#     5) Detect anomalies
#     6) Compare predicted anomalies with actual stock price data

# ### Preliminary Code

# In[1]:


#Importing the necessary packages
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Flatten
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from PIL import Image


# In[2]:


#Reading in the Pfizer stock price data
stock_prices = pd.read_csv('C:/Users/chink/Downloads/PFE.csv')
stock_prices


# This dataset contains dates ranging from June 1, 1972 to October 1st, 2021. It has 5 different stock prices for each date: "Open", "High", "Low", "Close", and "Adj Close". I will use the adjusted closing price as this quantity accounts for any corporate actions that affect the closing price of the stock, such as stock splits or rights offerings.

# In[3]:


#Visualizing prices
fig = px.line(stock_prices, x='Date', y='Adj Close',)
fig.update_layout(
    title="Pfizer Stock Price",
    xaxis_title="Date",
    yaxis_title="Adjusted Closing Price ($)")
fig.show()


# In[4]:


#Checking for null entries in the "Date" column
if np.sum(stock_prices['Date'].isnull()) == 0:
    print("There are no null entries in the Date column")
else:
    print("There are null entries in the Date column")


# In[5]:


#Checking for null entries in the "Adj Close" column
if np.sum(stock_prices['Date'].isnull()) == 0:
    print("There are no null entries in the Adj Close column")
else:
    print("There are null entries in the Adj Close column")


# In[ ]:





# ### Splitting and scaling the data

# To split the data, I will create a training and a test set. The test set consists of stock prices from the beginning January 2020 to the end September 2021. In creating the training set, I aim to choose subset of the data during which time I assume that there are almost zero anomalies. While there are many ways to construct this dataset, I will choose the decade of the 2010s as the training data, as this constitutes a large chunk of time and immediately precedes the test data.

# In[6]:


#Creating the training set
start_Jan_2010_ind = stock_prices[stock_prices['Date'].str.contains('2010-01')].index[0]
end_Dec_2019_ind = stock_prices[stock_prices['Date'].str.contains('2019-12')].index[-1]
training_data = stock_prices.loc[start_Jan_2010_ind:end_Dec_2019_ind][['Date','Adj Close']]


# Just a small note about the code chunk up above: due to weekends and holidays, the beginning date for Jan 2010 isn't just '2010-01-01' and likewise the end date of 2019 isn't just '2019-12-31'; for this reason, I need to collect the indices corresponding to the start and end dates
# for the period of time spanning the training set

# In[7]:


#Viewing the training set
training_data


# In[8]:


#Creating the test set in a similar fashion
start_Jan_2020_ind = stock_prices[stock_prices['Date'].str.contains('2020-01')].index[0]
end_Sep_2021_ind = stock_prices[stock_prices['Date'].str.contains('2021-09')].index[-1]
test_data = stock_prices.loc[start_Jan_2020_ind:end_Sep_2021_ind][['Date','Adj Close']]


# In[9]:


#Viewing the test data
test_data


# In[10]:


#Scaling the data
scale = MinMaxScaler()
scale.fit(training_data[['Adj Close']])
train = training_data.copy()
test = test_data.copy()
train['Adj Close'] = scale.transform(training_data[['Adj Close']])
test['Adj Close'] = scale.transform(test_data[['Adj Close']])


# In[ ]:





# ### Creating the time-series sequences

# The input fed into the LSTM model consists of sequences. The sequences have length t, where t is the equivalent to the time-step that is walked forward. For this model, the time step is set to 30, meaning that the sequences fed into the model correspond to month-long periods of time.

# In[11]:


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
    newshape = (len(input_seq),t_step,1)
    input_seq = np.reshape(input_seq,newshape)
    input_seq = np.array(input_seq)
    output_val = np.array(output_val)
    return input_seq,output_val


# Just a note about the code chunk above: for the LSTM model, the sequences must be 3D tensors of shape (training data length, time step, 1).

# In[12]:


#Creating the train sequences, predicted train stock prices, test sequences, and predicted test sequences
time_step = 30
train_sequences, train_sp = sequences(train,time_step)
test_sequences, test_sp = sequences(test,time_step)


# In[13]:


#Confirming the shapes of the datasets are correct
print("Training sequence data shape: ",train_sequences.shape)
print('Training stock price data shape: ',train_sp.shape)
print('Testing sequence data shape: ',test_sequences.shape)
print('Testing stock price data shape: ',test_sp.shape)


# In[ ]:





# ### Building the model

# I will build a model similar to that of @TareqTayeh, referenced earlier. In parallel with this notebook, I wrote a script using the package GridSearchCV to optimize the hyperparameters of the LSTM model with respect to the parameters of epochs, batch_size, and dropout rate. This and its output are uploaded in the repository for this project. For the autoencoder, I will use the parameters output by GridSearchCV 

# In[14]:


#Setting a seed for reproducibility
np.random.seed(2021)
tf.random.set_seed(2021)


# In[15]:


#Creating the model
model = Sequential()
model.add(LSTM(100, activation = 'tanh', input_shape=(train_sequences.shape[1],train_sequences.shape[2])))
model.add(Dropout(rate=0.15))
model.add(RepeatVector(train_sequences.shape[1]))
model.add(LSTM(100, activation = 'tanh', return_sequences=True))
model.add(Dropout(rate=0.25))
model.add(TimeDistributed(Dense(train_sequences.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()


# In[ ]:





# ### Training the model

# In[16]:


#Training the model on the sequence data
trained_model = model.fit(train_sequences, train_sequences, epochs=150, batch_size=75, validation_split=0.2, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min')], shuffle=False)


# In[17]:


#Viewing the training and validation loss
plt.plot(trained_model.history['loss'], label='Training loss')
plt.plot(trained_model.history['val_loss'], label='Validation loss')
plt.title(label='Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()


# In[18]:


#Running the model on the test data
model.evaluate(test_sequences, test_sp)


# In[ ]:





# ### Detecting anomalies

# To detect anomalies, I will do the following:
#    
#     (1) Subtract the predicted stock prices for the training data from the actual stock prices to find loss
#     
#     (2) Derive an anomaly threshold value from the training data for the reconstruction loss
#     
#     (3) Subtract the predicted stock prices for the test data from the actual stock prices
#     
#     (4) Isolate all dates from the test data that have a loss greater than or equal to the previously derived threshold value

# In[19]:


#Finding training loss
pred_seq_train = model.predict(train_sequences, verbose=0)
training_loss = np.mean(np.abs(pred_seq_train - train_sequences), axis=1)

#Viewing the loss  with a histogram
fig = px.histogram(training_loss,nbins=40)
fig.update_layout(
    title="Training Reconstruction Loss",
    xaxis_title="Loss",
    yaxis_title="Count",
    showlegend=False
)
fig.show()


# In[20]:


#Calculating the threshold
quintile = 99
threshold = np.percentile(training_loss,quintile)
print("The loss threshold value is ",threshold)


# I have set a very rigid threshold loss value by setting a high quintile value of 99. My aim is not to find as many anamolies as possible but simply to identify true anamolies. Setting a rigid threshold value increases the chance that the identified anamolies actually are anamolies.

# In[21]:


#Finding the test loss
pred_seq_test = model.predict(test_sequences, verbose=0)
testing_loss = np.mean(np.abs(pred_seq_test - test_sequences), axis=1)

#Viewing the loss  with a histogram
fig = px.histogram(testing_loss,nbins=40)
fig.update_layout(
    title="Testing Reconstruction Loss",
    xaxis_title="Loss",
    yaxis_title="Count",
    showlegend=False
)
fig.show()


# In[22]:


#Finding the anamolies
df = test.copy()
df = test.iloc[30:]
df['Reconstruction Loss'] = testing_loss
anomalies = df[df['Reconstruction Loss']>threshold]
indices = list(anomalies.index)
indices.sort()
anomalies = anomalies.loc[indices]
anomalies


# The model has deteced 80 dates as anomalies. Many of these dates are adjacent in time, meaning that the detected anomalies actually span periods of times. The next code chunk will actually print the start and end dates of these periods of time.

# In[23]:


#Finding the start and end dates of the periods of time spanned by the anomalies
l = []
for i in range(len(indices)):
    if i != len(indices)-1:
        if indices[i] != indices[i-1]+1 and indices[i] == indices[i+1]-1: #start of series
            entry = []
            entry.append(indices[i])
        if indices[i] == indices[i-1]+1 and indices[i] != indices[i+1]-1: #end of series
            entry.append(indices[i])
            l.append(entry)
    if i == len(indices)-1 and indices[i] == indices[i-1]+1:
        entry.append(indices[i])
        l.append(entry)
dates = []
for i in l:
    sub_dates = []
    sub_dates.append(anomalies.loc[i[0]]['Date'])
    sub_dates.append(anomalies.loc[i[-1]]['Date'])
    dates.append(sub_dates)
print('The anomalies span the following dates ')
for i in dates:
    print('From\t',i[0],'\tto\t',i[1])


# In[ ]:





# ### Comparing the predicted anomalies with the real-world stock data

# I will use Yahoo Finance's data on the Pfizer stock to check the accuracy of the predicted anomalies. In each of the charts displayed below, the purple line represents the 30-day moving average, which was selected because the time-step for the model was set to 30. The bars at the bottom indicate the volume traded each day, with red and green indicating increases or decreases in stock price, respectively.
# 
# The periods of time highlighted in light red represent represent the anomalies detected by the LSTM model. The dates highlighted in light blue indicate days with high volume. The idea behind taking notice of the volume bars is that large volumes can signify high buying or selling pressure, which drives large increases and decreases in stock price (respectively) and causes anomalies in the stock price as a result. For the purposes of this notebook, I will assume large volume changes to be indicative of anomalies to qualitatively assess the performance of the model.

# Jan to April 2020
# ![Jan_Apr_2020-2.png](attachment:Jan_Apr_2020-2.png)

# May to August 2020
# ![May_August_2020.png](attachment:May_August_2020.png)

# September to December 2020
# ![Sept_Dec_2020.png](attachment:Sept_Dec_2020.png)

# Jan to April 2021
# ![Jan_Apr_2021.png](attachment:Jan_Apr_2021.png)

# May to September 2021
# ![May_Sept_2021.png](attachment:May_Sept_2021.png)

# It can be seen from the charts above that in most of the time periods identified by the model as anomalies, there do exist large changes in trading volume. An exception to this the period of time from 12/09/20 to 12/14/20. Throughout 2020 and 2021, the autoencoder does seem to miss large changes in trading volume as well; some of the missed anomalies have trading volumes that outweigh those of the anomalies detected by the model (i.e. Sep-Dec 2020). Additionally, it is important to note that while the model does correctly predict periods of time in which there are anomalies, not every day in this period of time is an anomaly. All in all, in the context of forecasting the model shows mediocre performance. It may be useful in identifying periods of time during which one will observe anomalies, but it is not effective in predicting every anomaly or the largest anomalies during a period of time. More investigation is required to make this LSTM model more effective.

# In[ ]:





# ### References

#     1. https://github.com/TareqTayeh/Price-TimeSeries-Anomaly-Detection-with-LSTM-Autoencoders-Keras/blob/master/code/Time%20Series%20of%20Price%20Anomaly%20Detection%20with%20LSTM%20Autoencoders%20(Keras).ipynb
#     
#     2. https://www.kaggle.com/varpit94/pfizer-stock-data
#     
#     3. https://machinelearningmastery.com/lstm-autoencoders/
#     
#     4. https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
#     
#     5. https://finance.yahoo.com/quote/PFE?p=PFE&.tsrc=fin-srch
