from stock_brain import QLearningTable
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



price0=pd.read_csv('parameter88.csv')

if __name__=="__main__":
    RL=QLearningTable(actions=['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'])

protfolio_value=np.zeros(3)#change1
def transactionCost(pv, pv_, a, a_, p, p_, n, rate): #transaction cost of one stock
    # pv : portfolio value of time 1
    # pv_ : portfolio value of time 2
    # a : action(weight) of time 1
    # a_ : action(weight) of time 2
    # p : price of stock in time 1
    # p_ : price of stock in time 2
    # n : time
    # rate : transaction cost rate
    cost = abs(pv * a / p - pv_ * a_ / p_) * p_ * rate
    return cost

def benchmark(t, budget, price0, rate):
    # t : the interval of time transaction occurs
    # budget
    # price0 : price data
    # rate : transaction cost rate
        n = 1
        history = np.zeros(len(price0))
        history[0] = budget
        history[1]=budget
        pv = budget #portfolio value
        p = price0.iloc[0,2]  #p : price of stock 1 in the beginning
        _p = price0.iloc[0,4]   #_p : price of stock 2 in the beginning
        if t == 0: #no transaction
            while n < len(price0):
                s1_p = price0.iloc[n,2]
                s2_p = price0.iloc[n,4]
                pv = budget * 0.5 / p * s1_p + budget * 0.5 / _p * s2_p
                history[n] = pv
                n += 1
            return history

        else:    
            while n < len(price0):
                s1_p = price0.iloc[n,2]
                s2_p = price0.iloc[n,4]

                pv = budget * 0.5 / p * s1_p + budget * 0.5 / _p * s2_p
            
                if n % t == 0:
                    s1_pri = price0.iloc[n-t,2] #prior bought price
                    s2_pri = price0.iloc[n-t,4] #prior bought price
                    transaction_cost = abs(pv / s1_p - budget / s1_pri) * 0.5 * s1_p * rate + \
                                    abs(pv / s2_p - budget / s2_pri) * 0.5 * s2_p * rate
                    pv = pv - transaction_cost
                    p = s1_p
                    _p = s2_p
                    budget = pv
                history[n] = pv
                n += 1
            return history
    
for episode in range(3):#change2
    w=np.zeros(len(price0))#store the portfolio value of each state in this array w
    v=np.zeros(len(price0))#each step's bench mark (50%,50%)
    a0=np.zeros(len(price0))#store each action we chose in this array a0
    m=np.arange(0,len(price0),1)
    mm=np.arange(1,len(price0),1)#use it as the x-axis in the plot
    ep=np.arange(1,4,1)#change3
    #a0[0]=0.5
    #add=np.zeros(len(price0)-1)
    rate=np.zeros(len(price0)-1)
    rateb=np.zeros(len(price0)-1)
    rateb1=np.zeros(len(price0)-1)
    rewardt=[]
    rewardb=[]
    rewardt.append(0)
    rewardb.append(0)
    vt=[]
    vt.append(0)
    w[0]=10000#the beginning portfolio value is 10000
    v[0]=10000
    v[1]=10000#the beginning portfolio value is 10000
    n=0
    trate=0.002
    print(str(episode+1)+'/5000')#print out the number of episode
    while n==0:
        s=price0.parameter[0]
        s_=price0.parameter[1]
        s_!='terminal'
        a1=RL.choose_action(s)
        a=float(a1)
        a0[n]=a
        cost=w[n]*trate/(1+trate)
        #cost=abs((0-a)/(1-a*trate))*w[n]*trate+abs(0-(1-a))/(1-(1-a)*trate))*w[n]*trate
        w[n+1]=(a*price0.price1[1]/price0.price1[0]+(1-a)*price0.price2[1]/price0.price2[0])*(w[n]-cost)
        rate[0]=(w[n+1]-w[n])/w[n]
        #add[0]=w[n+1]-w[n]
        n=n+1
    while n!=0 and n<len(price0)-1:
        s=price0.parameter[n]
        s_=price0.parameter[n+1]
        s_!='terminal'
        a1=RL.choose_action(s)
        a=float(a1)
        a0[n]=a
        #reward=(a*price0.mean1[n+1]+(1-a)*price0.mean2[n+1])/(a*price0.sd1[n+1]+(1-a)*price0.sd2[n+1])
        #v0=(price0.mean1[n+1]+price0.mean2[n+1])/(price0.sd1[n+1]+price0.sd2[n+1])
        #vt.append(v0)
        if w[n-1]*a0[n-1]*price0.price1[n]>w[n]*a*price0.price1[n-1]:
            cost=(price0.price1[n]*w[n-1]*a0[n-1]*trate/price0.price1[n-1]-w[n]*a*trate)/(1/2-a*trate)
        elif w[n-1]*a0[n-1]*price0.price1[n]<w[n]*a*price0.price1[n-1]:
            cost=(w[n]*a*trate-price0.price1[n]*w[n-1]*a0[n-1]*trate/price0.price1[n-1])/(1/2+a*trate)
        #cost=abs((a0[n-1]-a)/(1-a*trate))*w[n]*2*trate
        reward0=a*price0.price1[n+1]/price0.price1[n]+(1-a)*price0.price2[n+1]/price0.price2[n]
        w[n+1]=(w[n]-cost)*reward0
        #add[n]=w[n+1]-w[n]
        rate[n]=(w[n+1]-w[n])/w[n]
        reward=np.mean(rate[n-1:n+1])/np.std(rate[n-1:n+1])
        rewardt.append(reward)
        if w[n-1]*price0.price1[n]>w[n]*price0.price1[n-1]:
            bcost=(price0.price1[n]*w[n-1]*trate/price0.price1[n-1]-w[n]*trate)/(1-trate)
        elif w[n-1]*price0.price1[n]<w[n]*price0.price1[n-1]:
            bcost=(w[n]*trate-price0.price1[n]*w[n-1]*trate/price0.price1[n-1])/(1+trate)
        #bcost=abs((1-trate)/(2-trate))*v[n]#transaction cost for benchmark
        v[n+1]=(price0.price1[n+1]/price0.price1[n]+price0.price2[n+1]/price0.price2[n])*(v[n]-bcost)/2
        rateb[n]=(v[n+1]-v[n])/v[n]
        if rateb[n]>=0:
            rateb1[n]=0
        else:
            rateb1[n]=rateb[n]
        rewardbench=np.mean(rateb[n-1:n+1])/np.std(rateb1[n-1:n+1])
        rewardb.append(rewardbench)
        RL.learn(s,a1,reward,s_)
        n=n+1  
    while n==len(price0)-1:
        s_=str('terminal')
        a1=RL.choose_action(s)
        a=float(a1)
        a0[n]=a
        s=s_
        n=n+1
    protfolio_value[episode]=w[len(price0)-2]

#v=benchmark(10,10000,price0,0.02)
print(w)
print(len(m))
print(len(v))
print(len(rewardt))
print(RL.q_table)
print(v)
RL.q_table.to_csv('q_tablePARAMETER88.csv',index=True)
#RL.q_table.to_csv('q_tablefixingtransaction10.csv',index=True)
plt.plot(m,w,label='q-learning')
plt.plot(m,v,label='benchmark(0.5,0.5)')
plt.legend()
plt.show()
plt.plot(mm,rewardt,label='sharpe')
plt.plot(mm,rewardb,label='banch')
plt.legend()
plt.show()
#plt.plot(m,rewardt,label='sharpe')
#plt.plot(ep,protfolio_value)
#plt.show()
