# DMNN 
### Dynamic Modification Neural Network Model for Short-term Traffic Prediction






Abstract—Short-term precise prediction of network traffic can effectively help operators reasonably allocate network resources, improve network service quality and scheduling efficiency. Our model proposed in this paper uses dynamic parameters to adjust the predicted value and refit the predicted sequence for short-term traffic prediction which is called Dynamic Modification Neural Network (DMNN) model. The architecture of our model consists of a prediction module, an adjustment module and a series re-fitting module. The prediction module learns time closeness of the traffic data before the predicted point. The adjustment module takes dynamic period characteristics of data into account and generates adjusted value by linear discrete dynamic parameters. The series re-fitting module refits predicted series from the first two modules with a new hybrid loss function. In this paper, we take different neural networks in the prediction module, such as Long Short-Term Memory (LSTM), Bi-directional Long Short-Term Memory (BiLSTM), Convolutional Long Short-Term Memory (ConvLSTM) and Convolutional Bi-directional Long Short-Term Memory (ConvBiLSTM). We evaluate the performance of the proposed model by using two real-world datasets for short-term traffic flow prediction. Moreover, Experimental results show that the proposed model with dynamic modification has much higher accuracy than other models and decreases the prediction value error and time skew of the inflection points.


### Need to unzip the .zip file.

Update models for traffic prediction 

Now model.py contains MLP, LSTM, BiLSTM, conv_BiLSTM, LSTM_with_attention, Conv_LSTM_with_attention. 

ALL codes are coded by Python 3.6 and Pytorch 1.3  
