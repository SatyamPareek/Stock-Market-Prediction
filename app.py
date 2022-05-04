import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


st.title('Stock Price Predictor')

option = st.selectbox(
     'Select the Company Name',
     ('Google', 'Wipro'))


if(option=='Google'):
    df_train=pd.read_csv('GOOG_train.csv')
    df_test=pd.read_csv("GOOG_test.csv")


    st.subheader('Opening Price vs Time (2005-2017)')
    fig=plt.figure(figsize=(12,6))
    plt.plot(df_train.Open)
    st.pyplot(fig)

    features=list(df_train)[1:8]
    df_for_training=df_train[features].astype(float)
    df_for_testing=df_test[features].astype(float)


    #Scaling the data
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    df_for_training_scaled=scaler.fit_transform(df_for_training)


    #Load The Model
    model=load_model('stock.h5')

    past_30_days=df_for_training.tail(30)
    final_testing_df=past_30_days.append(df_for_testing,ignore_index=True)
    
    #now scale the testing data
    input_data=scaler.fit_transform(final_testing_df)

    x_test=[]
    y_test=[]

    for i in range(30,input_data.shape[0]):
        x_test.append(input_data[i-30:i])
        y_test.append(input_data[i,0])

    
    x_test,y_test=np.array(x_test),np.array(y_test)

    #Making Predictons
    y_pred=model.predict(x_test)

    prediction_copies = np.repeat(y_pred, df_for_training.shape[1], axis=-1)
    y_pred = scaler.inverse_transform(prediction_copies)[:,0]

    y_test=y_test.reshape(-1,1)
    testing_copies=np.repeat(y_test,df_for_training.shape[1],axis=-1)
    y_test=scaler.inverse_transform(testing_copies)[:,0]




    st.subheader('Predictons vs Original (2018-2022)')
    fig=plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original Price')
    plt.plot(y_pred,'r',label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)




elif option=='Wipro':
    df_train=pd.read_csv('WIPRO_train.csv')
    df_test=pd.read_csv("WIPRO_test.csv")


    st.subheader('Opening Price vs Time (2005-2017)')
    fig=plt.figure(figsize=(12,6))
    plt.plot(df_train.Open)
    st.pyplot(fig)

    features=list(df_train)[1:8]
    df_for_training=df_train[features].astype(float)
    df_for_testing=df_test[features].astype(float)


    #Scaling the data
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    df_for_training_scaled=scaler.fit_transform(df_for_training)


    #Load The Model
    model=load_model('stock_wipro.h5')

    past_30_days=df_for_training.tail(30)
    final_testing_df=past_30_days.append(df_for_testing,ignore_index=True)
    
    #now scale the testing data
    input_data=scaler.fit_transform(final_testing_df)

    x_test=[]
    y_test=[]

    for i in range(30,input_data.shape[0]):
        x_test.append(input_data[i-30:i])
        y_test.append(input_data[i,0])

    
    x_test,y_test=np.array(x_test),np.array(y_test)

    #Making Predictons
    y_pred=model.predict(x_test)

    prediction_copies = np.repeat(y_pred, df_for_training.shape[1], axis=-1)
    y_pred = scaler.inverse_transform(prediction_copies)[:,0]

    y_test=y_test.reshape(-1,1)
    testing_copies=np.repeat(y_test,df_for_training.shape[1],axis=-1)
    y_test=scaler.inverse_transform(testing_copies)[:,0]




    st.subheader('Predictons vs Original')
    fig=plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original Price')
    plt.plot(y_pred,'r',label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

