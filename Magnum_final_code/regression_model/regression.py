import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
price0=pd.read_csv('observationAmatCaj1y.csv')
price0.columns=pd.Series(['Date','Open1','Close1','Open2','Close2'])
n=0
t=10
parameter=[]
#mean1=[]
#sd1=[]
#mean2=[]
#sd2=[]
#price1r=[]
#price2r=[]
price1=price0.Close1[t-1:len(price0)][:,np.newaxis]
price2=price0.Close2[t-1:len(price0)][:,np.newaxis]
while len(price0)+1>n+t:
    y1=price0.Close1[n:n+t]
    y2=price0.Close2[n:n+t]
    x=np.arange(1,t+1,1)
    slope1, intercept1, r_value1, p_value1, slope_std_error1 = stats.linregress(x, y1)
    slope2, intercept2, r_value2, p_value2, slope_std_error2 = stats.linregress(x, y2)
    slope=','.join([str(round(slope1,1)),str(round(slope2,1))])
    parameter.append(slope)
    #price1r.append((price1[n+1]-price1[n])/price1[n])
    #price2r.append((price2[n+1]-price2[n])/price2[n])
    #mean1.append(np.mean(y1))
    #sd1.append(np.std(y1))
    #mean2.append(np.mean(y2))
    #sd2.append(np.std(y2))
    n=n+1
print(parameter)
parameter=np.array(parameter)[:,np.newaxis]
#mean1=np.array(mean1)[:,np.newaxis]
#mean2=np.array(mean2)[:,np.newaxis]
#sd1=np.array(sd1)[:,np.newaxis]
#sd2=np.array(sd2)[:,np.newaxis]

print(price1.shape)
print(np.array(parameter).shape)
csv=np.hstack((parameter,price1,price2))
csv=pd.DataFrame(csv)
csv.columns=pd.Series(['parameter','price1','price2'])
csv.to_csv('parameter_testAC10.csv',index=True)
#y=df.Close1
#x=df.Day
#x=sm.add_constant(x)
#est=sm.OLS(y,x)
#est=est.fit()
##print(est.summary())
##print(est.params)
#parameter=[]
#parameter.append(est.params)
##print(parameter)
#print(round(parameter[0],2))
