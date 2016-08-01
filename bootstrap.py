import numpy as np 
import random 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 

def rolling_window(a,window):
	shape=a.shape[:-1]+(a.shape[-1]-window+1,window)
	strides=a.strides+(a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)

days,raw_gcc,fit_gcc=np.loadtxt('Time_series_data.csv',delimiter=',',skiprows=1,unpack=True,dtype=[('days',np.int32),('raw_gcc',np.float32),('fit_gcc',np.float32)]) #Data extraction

fig=plt.figure()    
ax1=plt.subplot2grid((6,2),(0,0),rowspan=2,colspan=1)  
# ax.scatter(raw_gcc,fit_gcc)
ax1.plot(raw_gcc) #Plot the raw_gcc data
ax1.set_xlabel('Time')
ax1.set_title('Rolling mean & data') 
diff=np.diff(raw_gcc)
# diff = np.insert(diff,0,0)

# diff_lag=diff[1:]
# diff_lag=np.append(diff_lag,0)
# print (diff[:10])
# print (diff_lag[:10])
# ax.scatter(diff,diff_lag, label='Diff')
# ax.set_xlabel('Lag')
# ax.set_ylabel('DIFF')

window=30
day_roll=rolling_window(days,window)
raw_gcc_roll=rolling_window(raw_gcc,window)  #Rolling 30 data points
fit_gcc_roll=rolling_window(fit_gcc,window)  #Rolling 30 data points
mean_array=np.mean(raw_gcc_roll,axis=1)  #Mean of the rolling data
ax1.plot(mean_array)
ax2=plt.subplot2grid((6,2),(0,1),rowspan=2,colspan=1)
std_array=np.std(raw_gcc_roll,axis=1)   #Standard deviation of the rolling data
ax2.plot(raw_gcc)
ax2.plot(std_array)
ax2.set_title('Rolling std. deviation & data')
ax2.set_xlabel('Time')


#Perform Dickey-Fullr test
dft=sm.tsa.stattools.adfuller(raw_gcc)
print dft
ax3=plt.subplot2grid((6,2),(2,0),rowspan=2,colspan=1)
fig=sm.graphics.tsa.plot_acf(raw_gcc,ax=ax3,alpha=0.05) #Auto correlation function of raw_gcc

ax4=plt.subplot2grid((6,2),(2,1),rowspan=2,colspan=1)
fig=sm.graphics.tsa.plot_pacf(raw_gcc,ax=ax4,alpha=0.05) #Partial Auto Correlation function of raw_gcc

seas=sm.tsa.seasonal_decompose(raw_gcc,freq=12)
ax5=plt.subplot2grid((6,2),(4,0),rowspan=2,colspan=1)
seas.plot()    #Plot of the original, trend, seasonal, residual aspect of the time series

# res=sm.tsa.arma_order_select_ic(raw_gcc,ic=['aic','bic'],trend='nc',fit_kw={})
# print res
# fig=plt.figure(figsize=(12,8))
# ax1=fig.add_subplot(211)
# fig=sm.graphics.tsa.plot_acf(diff,ax=ax1)
# ax2=fig.add_subplot(212)
# fig=sm.graphics.tsa.plot_pacf(diff,ax=ax2)   #From the plot, it can seen that PACF shuts off after 3 lages. Hence, order of the AR model is 3
plt.show() 	
# arma_mod10=sm.tsa.ARMA(diff,(1,0)).fit()
# print arma_mod10.params

# arma_mod20=sm.tsa.ARMA(diff,(2,0)).fit()
# print arma_mod20.params

# arma_mod11=sm.tsa.ARMA(diff,(1,1)).fit()
# print arma_mod11.params

# arma_mod22=sm.tsa.ARMA(diff,(2,2)).fit()
# print arma_mod22.params

# arma_mod30=sm.tsa.ARMA(fit_gcc,(1,0)).fit()
# print arma_mod30.params
