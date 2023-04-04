import numpy as np
import numpy.random 
import math
import warnings
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split #训练测试未来走势模块
####################################
# S=265881.04
# X=991.38
# r=0.0397
# sigma=0.3131
# T=5

S=9.15
X=9.28
r=0.0005
sigma=0.0336
T=28
q=0.000155#股利支付率

I=5000#path
M=4000#number of prices
dt=T/M
####################################
def reg():
    x=np.array([1,2,3,4,5,6])
    y=np.array([2.3,3.5,4.2,5.1,6.5,8.3])
    corr=np.polyfit(x,y,6)#二阶函数 y=aX^2+bx+c 以此类推
    Y=np.poly1d(corr)#最小二乘法函数形态 
    x_li=np.arange(20)
    print('Y=\n%s'%Y)
    print('当函数自变量=%s时,y=%.4f'%(6,Y(6)))#函数拟合值
    print(Y(x_li))#拟合
def quadratic_fitting(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    S0,S1,S2,S3,S4 = len(X),sum(X),sum(X*X),sum(X**3),sum(X**4)
    V0,V1,V2 = sum(Y), sum(Y*X), sum(Y*X*X)
    coeff_mat = np.array([[S0,S1,S2],[S1,S2,S3],[S2,S3,S4]])#2阶函数
    target_vec = np.array([V0,V1,V2])
    inv_coeff_mat = np.linalg.inv(coeff_mat)#逆矩阵
    fitted_coeff = np.matmul(inv_coeff_mat,target_vec)#矩阵相乘
    resulted_Ys = fitted_coeff[0]+fitted_coeff[1]*X+fitted_coeff[2]*X*X
    return resulted_Ys
def reg2(x,y,n=2):
    corr=np.polyfit(x,y,n)
    Y=np.poly1d(corr)
    # print('Y',y)
    # print('pred',Y(x))
    return Y(x)
def reg3(x,y):#statsmodels循环内存错误
    x_1=np.exp(-0.5*x)
    x_2=x_1*(1-x)
    x_3=x_1*(1-2*x+x*x/2)
    X=np.zeros(shape=(4,len(x)))
    X[0]=x_1;X[1]=x_2;X[2]=x_3;X[3]=y
    data=pd.DataFrame(X.T,columns=('X','XX','XXX','y'),dtype=float)
    # ols=sm.OLS.from_formula('y ~ X+XX+XXX -1',data=data).fit()#sm.formula.ols()带截距
    ols=sm.formula.ols('y ~ X+XX+XXX',data=data).fit()
    pred=ols.predict(data[['X','XX','XXX']])
    Y=np.array(pred.tolist())
    return Y
def reg4(x,y):
    x_1=np.exp(-0.5*x)
    x_2=x_1*(1-x)
    x_3=x_1*(1-2*x+x*x/2)
    X=np.zeros(shape=(4,len(x)))
    X[0]=x_1;X[1]=x_2;X[2]=x_3;X[3]=y
    data=pd.DataFrame(X.T,columns=('X','XX','XXX','y'),dtype=float)
    xinput=data[['X','XX','XXX']]
    yinput=data[['y']]
    Y=LinearRegression(fit_intercept=0)#fit_intercept=0
    Y.fit(xinput,yinput)#Y.intercept_截距，Y.coef_回归系数，print
    pred=Y.predict(xinput)
    return pred.T
def lookback_min_array(martix):#得到矩阵中每一列最小的值
    res_array=[]
    for j in range(len(martix[0])):
        one_array=[]
        for i in range(len(martix)):
            one_array.append(int(martix[i][j]))
        res_array.append(min(one_array))
    return res_array    
def lookback_max_array(martix):#得到矩阵中每一列最大的值
    res_array=[]
    for j in range(len(martix[0])):
        one_array=[]
        for i in range(len(martix)):
            one_array.append(int(martix[i][j]))
        res_array.append(max(one_array))
    return res_array        
####################################
def gen_randompath(s=S,r=r,sigma=sigma,t=T,m=M,i=I):
    global dt
    standard=np.random.standard_normal((m+1,I))
    pricepath=np.zeros((m+1,I))
    pricepath[0]=s
    for n in range(1,m+1):
        pricepath[n]=pricepath[n-1]*np.exp((r-0.5*sigma**2)*dt+sigma*math.sqrt(dt)*standard[n])
    return pricepath
def gen_randompath_q(s=S,r=r,sigma=sigma,t=T,m=M,i=I):#含分红
    global dt,q
    standard=np.random.standard_normal((m+1,I))
    pricepath=np.zeros((m+1,I))
    pricepath[0]=s
    for n in range(1,m+1):
        pricepath[n]=pricepath[n-1]*np.exp((r-q-0.5*sigma**2)*dt+sigma*math.sqrt(dt)*standard[n])
    return pricepath
def lsm_call(path,K=X):
    global dt,M,r
    option_prices=np.maximum(path[-1,:]-K,0)
    for i in range(M-1,0,-1):
        option_prices*=np.exp(-r*dt)
        # option_prices=quadratic_fitting(path[i,:],option_prices)
        option_prices=reg2(path[i,:],option_prices,5)
        # option_prices=reg3(path[i,:],option_prices)
        # option_prices=reg4(path[i,:],option_prices)
        option_prices = np.maximum(option_prices,path[i,:]-K)
    option_prices *= np.exp(-r*dt)
    return option_prices.mean()
def lsm_put(path,K=X):
    global dt,M,r
    option_prices=np.maximum(K-path[-1,:],0)
    for i in range(M-1,0,-1):
        option_prices *= np.exp(-r*dt)
        option_prices = reg2(path[i,:], option_prices,5)
        option_prices = np.maximum(option_prices,K-path[i,:])
    option_prices *= np.exp(-r*dt)
    return option_prices.mean()
def EUR_call(path,K=X):
    global dt,M,r,T
    option_prices=np.exp(-r*T)*np.maximum(path[-1,:]-K,0)
    return option_prices.mean()
def EUR_put(path,K=X):
    global dt,M,r,T
    option_prices=np.exp(-r*T)*np.maximum(K-path[-1,:],0)
    return option_prices.mean()
def lookbackoption_EUR_call(path,K=X):
    global dt,M,r,T
    Pmax=lookback_max_array(path)
    option_prices=np.exp(-r*T)*np.maximum(np.array(Pmax)-K,0)
    return option_prices.mean()
def lookbackoption_EUR_put(path,K=X):
    global dt,M,r,T
    Pmin=lookback_min_array(path)
    option_prices=np.exp(-r*T)*np.maximum(K-np.array(Pmin),0)
    return option_prices.mean()
def Asian_call(path,K=X):
    global dt,M,r,T
    option_prices=np.exp(-r*T)*np.maximum(np.mean(path,axis=0)-K,0)
    return option_prices.mean()
def Asian_put(path,K=X):
    global dt,M,r,T
    option_prices=np.exp(-r*T)*np.maximum(K-np.mean(path,axis=0),0)
    return option_prices.mean()
if __name__=='__main__':
    with np.errstate(divide='ignore'):
        np.float64(1.0)/0.0
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    warnings.filterwarnings('ignore')
    # reg()
    
    path=gen_randompath_q()
    AMer_c=lsm_call(path)
    AMer_p=lsm_put(path)
    EUR_c=EUR_call(path)
    EUR_p=EUR_put(path)
    lookback_c=lookbackoption_EUR_call(path)
    lookback_p=lookbackoption_EUR_put(path)
    Asian_c=Asian_call(path)
    Asian_p=Asian_put(path)

    # if 1:
    #     if 1:
    #         plt.figure(figsize=(10,8))
    #         plt.hist(path[-1],bins=50)
    #         plt.xlabel('到期日')
    #         plt.ylabel('频率')
    #         plt.title('股价分布')
    #         plt.show()
    #     else:
    #         plt.figure(figsize=(10,8))
    #         plt.plot(path[:200],lw=1.5)
    #         plt.xlabel('时间')
    #         plt.ylabel('股票价格')
    #         plt.title('股价轨迹')
    #         plt.show()
#折扣率 期权/起始价格
    print(
        '美C：%s,美P：%s；欧C：%s，欧P：%s；欧回C：%s，欧回P：%s；亚C：%s，亚P：%s\n美折：%s，欧折：%s，回望折：%s，亚折：%s'
        %(AMer_c,AMer_p,EUR_c,EUR_p,lookback_c,lookback_p,Asian_c,Asian_p,AMer_p/S,EUR_p/S,lookback_p/S,Asian_p/S)
          )