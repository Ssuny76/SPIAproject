import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stock_brain import QLearningTable
if __name__=="__main__":
    RL=QLearningTable(actions=['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'])

q_table = pd.read_csv('q_tablefixingtransaction10y.csv',index_col = 0)
price0 = pd.read_csv('parameter_testAC.csv')
w=np.zeros(len(price0))
v=np.zeros(len(price0))
a0=np.zeros(len(price0))
m=np.arange(0,len(price0),1)
mm=np.arange(1,len(price0),1)
a0[0]=0.5
rate=np.zeros(len(price0)-1)
rateb=np.zeros(len(price0)-1)
rewardt=[]
rewardb=[]
rewardt.append(0)
rewardb.append(0)
vt=[]
vt.append(0)
w[0]=10000
v[0]=10000
v[1]=10000
n=0
trate=0.002
ran=0
while n==0:
    s=price0.parameter[0]
    s_=price0.parameter[1]
    s_!='terminal'
    if s in q_table.index:
        action=q_table.loc[s,:]
        action=float(np.argmax(action))
        
    else:
        action=0.5
        #action = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        ran=ran+1
    a0[n] = float(action)
    cost=w[n]*trate/(1+trate)
    w[n+1]=(action*price0.price1[1]/price0.price1[0]+(1-action)*price0.price2[1]/price0.price2[0])*(w[n]-cost)
    rate[0]=(w[n+1]-w[n])/w[n]
    n=n+1
while n!=0 and n<len(price0)-1:
    s=price0.parameter[n]
    s_=price0.parameter[n+1]
    s_!='terminal'
    if s in q_table.index:
        action=q_table.loc[s,:]
        action=float(np.argmax(action))
        
    else:
        action=w[n-1]*a0[n-1]*price0.price1[n]/(w[n]*price0.price1[n-1])
        #action=a0[n-1]
        #action=0.5
        #action = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        ran=ran+1
    a0[n] = float(action)
    if w[n-1]*a0[n-1]*price0.price1[n]>w[n]*action*price0.price1[n-1]:
        cost=(price0.price1[n]*w[n-1]*a0[n-1]*trate/price0.price1[n-1]-w[n]*action*trate)/(1/2-action*trate)
    elif w[n-1]*a0[n-1]*price0.price1[n]<w[n]*action*price0.price1[n-1]:
        cost=(w[n]*action*trate-price0.price1[n]*w[n-1]*a0[n-1]*trate/price0.price1[n-1])/(1/2+action*trate)
    else:
        cost=0

    reward0=action*price0.price1[n+1]/price0.price1[n]+(1-action)*price0.price2[n+1]/price0.price2[n]
    #w[n+1]=(w[n]-cost)*reward0
    w[n+1]=w[n]*reward0-cost
    rate[n]=(w[n+1]-w[n])/w[n]
    reward=np.mean(rate[n-1:n+1])/np.std(rate[n-1:n+1])
    rewardt.append(reward)
    if w[n-1]*price0.price1[n]>w[n]*price0.price1[n-1]:
        bcost=(price0.price1[n]*w[n-1]*trate/price0.price1[n-1]-w[n]*trate)/(1-trate)
    elif w[n-1]*price0.price1[n]<w[n]*price0.price1[n-1]:
        bcost=(w[n]*trate-price0.price1[n]*w[n-1]*trate/price0.price1[n-1])/(1+trate)

    v[n+1]=(price0.price1[n+1]/price0.price1[n]+price0.price2[n+1]/price0.price2[n])*(v[n]-bcost)/2
    rateb[n]=(v[n+1]-v[n])/v[n]
    rewardbench=np.mean(rateb[n-1:n+1])/np.std(rateb[n-1:n+1])
    rewardb.append(rewardbench)
    n=n+1
while n==len(price0)-1:
    s_=str('terminal')
    if s in q_table.index:
        action=q_table.loc[s,:]
        #print(action)
        action=float(np.argmax(action))
        #print(action)
        
    else:
        #action=w[n-1]*a0[n-1]*price0.price1[n]/(w[n]*price0.price1[n-1])
        #action=a0[n-1]
        #action=0.5
        action = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        ran=ran+1
    a0[n]=float(action)
    s=s_
    n=n+1
print(ran)
print(a0)
plt.plot(m,w,label='q-learning')
plt.plot(m,v,label='benchmark(0.5,0.5)')
plt.legend()
plt.show()
plt.plot(mm,rewardt,label='sharpe')
plt.plot(mm,rewardb,label='banch')
plt.legend()
plt.show()
    
