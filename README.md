# temperatureForecasting

This is the original source code used for all experiments in the paper "hort-Term Temperature Forecasts Using a Convolutional Neural Network – an Application to Different Weather Stations in Germany" in the Journal "Machine Learning and Knowledge Extraction" (MAKE)
TODO: link as soon as the paper has been accepted and published

This repository is organized as follows:

(a) Directories:
- 'data': contains the preprocessed data
- 'src\model': contains source code for all models (convLSTM and ARIMA)
- 'src\preprocess': contains data preprocessing methods
- 'src\vis': visualisation code

(b) Files:
- main.py: main-file for training the convLSTM network
- OnlyLSTM_NEW.py and OnlyLSTM_NEW_oneChannel.py: contains source code for the LSTM variant (without convolutional layers)
- visualize_results.py: visualisation stuff
