import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

##Preprocess
import src.config as cfg
import src.preprocess.Preprocess as pre
import src.vis.myVisualization as Visualization
import src.model.convLSTM.myConvLSTM as convLSTM

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

import seaborn as sns
from scipy import stats

hourstopredict = 24

cfg.prediction['num_predictions'] = hourstopredict # 6h
cfg.windowing['windowlength'] = 18
cfg.data['sequencelength'] =  24 #6*24


#%%##########################  
##### PREPROCESSING #########
############################# 
prep = pre.DataPreprocessing()
dataX, datay = prep.load_selected()


file_scaler = open('saver/outputs/scaler/scaler.pckl', 'wb') 
vis = Visualization.myVisualization('saver/Kempten/1000/outputs')
vis.load_saved_files(selected_data='test')

arima = False

#%% Visualize Loss
#vis.visualize_losses(50)
#
errNaiv, errArima, errNN, errNNLSTM,errNNLSTM2, x_after_list =  vis.visualizeTS(selected_dat='test',count = -1 ,arima=arima,visualTS = False)




   
naiv_std = np.std(np.array(errNaiv), axis=0)
if arima:
    arima_std = np.std(np.array(errArima), axis=0)
nn_std =np.std(np.array(errNN), axis=0)
nnLSTM_std =np.std(np.array(errNNLSTM), axis=0)
nnLSTM2_std = np.std(np.array(errNNLSTM2), axis=0)


xSize = 8
ySize = 5

# blue
a1, a2, a3 = 0/256, 85/256 , 163/256
# red 
b1, b2, b3 = 139/256, 0/256 , 0/256
# green
c1, c2, c3 = 34/256, 139/256 , 34/256
# plot std of all forecasts without gridSearch
plt.figure(figsize=(xSize,ySize))
plt.plot(np.array(x_after_list[-1])-np.array(x_after_list[-1])[0],
         naiv_std, 'g^-',linewidth = 2, markersize=8, color=(a1,a2,a3))
if arima:
    plt.plot(np.array(x_after_list[-1])-np.array(x_after_list[-1])[0],
             arima_std, 'bx-',linewidth = 2, markersize=8, color=(b1,b2,b3))
plt.plot(np.array(x_after_list[-1])-np.array(x_after_list[-1])[0],
         nn_std, 'ro-',linewidth = 2, markersize=8, color=(c1,c2,c3))
plt.plot(np.array(x_after_list[-1])-np.array(x_after_list[-1])[0],
         nnLSTM_std, 'ro-',linewidth = 2, markersize=8, color="m")
plt.plot(np.array(x_after_list[-1])-np.array(x_after_list[-1])[0],
         nnLSTM2_std, 'ro-',linewidth = 2, markersize=8, color="y")
plt.grid(True)
plt.xlabel(r'Zeit (h)')
plt.ylabel(r'Standardabweichung ($^\circ$C)')
if arima:
    plt.legend([r'SN', 
                r'SARIMA', 
                r'convLSTM',
                r'LSTM'])
else:
    plt.legend([r'SN', 
#            r'SARIMA', 
            r'convLSTM',
            r'LSTM (multichannel)', 
            r'LSTM (one channel)'])
    
print('Error ConvLSTM:', np.mean(errNN))
print('_________________________')
print('Error LSTM:', np.mean(errNNLSTM))

meanerrNN = np.mean(np.array(errNN), axis=0)
meanerrLSTM = np.mean(np.array(errNNLSTM), axis=0)
meanerrLSTM2 = np.mean(np.array(errNNLSTM2), axis=0)
meannaive = np.mean(np.array(errNaiv), axis=0)

plt.figure()
plt.plot(np.arange(24),np.abs(np.transpose(meanerrNN)), 'ro-',linewidth = 2, markersize=8, color=(c1,c2,c3))
plt.plot(np.arange(24),np.abs(np.transpose(meanerrLSTM)), 'ro-',linewidth = 2, markersize=8, color="m")
plt.plot(np.arange(24),np.abs(np.transpose(meanerrLSTM2)), 'ro-',linewidth = 2, markersize=8, color="y")
plt.plot(np.arange(24),np.abs(np.transpose(meannaive)), 'g^-',linewidth = 2, markersize=8, color=(a1,a2,a3))


plt.grid(True)
plt.xlabel(r'Zeit (h)')
plt.ylabel(r'mean error ($^\circ$C)')
plt.legend([r'convLSTM',
            r'LSTM (multichannel)', 
            r'LSTM (one channel)'])
#plt.savefig("solution/standardDeviation_24h.pdf", bbox_inches='tight')


#xSize = 6
#ySize = 4
## plot boxplot seasonal naiv
#df = pd.DataFrame()
#df = df.append(errNaiv)
#df.plot.box(grid='True', showfliers=False, figsize=(xSize,ySize))
#plt.xlabel(r'Zeit (h)')
#plt.ylabel(r'Temperatur ($^\circ$C)')
##plt.title("seasonal naiv")
##plt.savefig("solution/boxplot_seasonalNaiv_24h.pdf", bbox_inches='tight')
#
## plot boxplot SARIMA
#df = pd.DataFrame()
#df = df.append(errArima)
#df.plot.box(grid='True', showfliers=False, figsize=(xSize,ySize))
#plt.xlabel(r'Zeit (h)')
#plt.ylabel(r'Temperatur ($^\circ$C)')
#plt.show()
#plt.title("SARIMA")
#plt.savefig("solution/boxplot_SARIMA_24h.pdf", bbox_inches='tight')

# plot boxplot NN
#df = pd.DataFrame()
#df = df.append(errNN)
#df.plot.box(grid='True', showfliers=False, figsize=(xSize,ySize))
#plt.xlabel(r'Zeit (h)')
#plt.ylabel(r'Temperatur ($^\circ$C)')
##plt.title("NN")
#plt.savefig("solution/boxplot_NN_24h.pdf", bbox_inches='tight')
#
#
## density plots
#plt.figure(figsize=(xSize,ySize))
#sns.set(color_codes=True)
#ax = sns.distplot(np.mean(errNaiv, axis=0), norm_hist=False)
#ax.set(xlabel=r'mean error', ylabel=r'density')
#plt.savefig("solution/density_plot_naiv_24h.pdf", bbox_inches='tight')
#plt.clf()
#
#plt.figure(figsize=(xSize,ySize))
#sns.set(color_codes=True)
#ax = sns.distplot(np.mean(errArima, axis=0), norm_hist=False)
#ax.set(xlabel=r'mean error', ylabel=r'density')
#plt.savefig("solution/density_plot_arima_24h.pdf", bbox_inches='tight')
#plt.clf()
#
#plt.figure(figsize=(xSize,ySize))
#sns.set(color_codes=True)
#ax = sns.distplot(np.mean(errNN, axis=0), norm_hist=False)
#ax.set(xlabel=r'mean error', ylabel=r'density')
#plt.savefig("solution/density_plot_nn_24h.pdf", bbox_inches='tight')
