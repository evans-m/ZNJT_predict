import pandas as pd
from sklearn import svm
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import pickle
df=pd.read_pickle('DataMing_team/evans/zhjt_predict/cleand_df')
variables=['length', 'width','pr_full','weekday', 'hour', 'minute','month']
target=['travel_time']
##抽样
sample=df.sample(n=200000,random_state=0)
X=sample.loc[:,variables]
y=sample.loc[:,target]
##训练集测试集划分
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.4,random_state=1)
##按训练集标准化自变量
scaler=StandardScaler()
scaler.fit(X_train)
train_x=scaler.transform(X_train)
test_x=scaler.transform(X_test)
##训练SVR
clf1=svm.SVR()
clf1.fit(train_x,Y_train)
Y_predict1=clf1.predict(test_x)
##计算MAE
MAE_of_SVR=mean_absolute_error(Y_test,Y_predict1)
##保存模型
from sklearn.externals import joblib
joblib.dump(clf1,"SVR_1.pkl")
##训练多层感知机
clf2=MLPRegressor()
clf2.fit(train_x,Y_train)
Y_predict2=clf2.predict(test_x)
MAE_of_MLP=mean_absolute_error(Y_test,Y_predict2)
joblib.dump(clf2,"MLP_1.pkl")