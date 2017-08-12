#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
#Author: evans
#@Time: 2017/8/12 上午8:30


import pandas as pd
import numpy as np
import datetime
from datetime import timedelta

##定义评价函数

def mape_object(y,d):
    g=1.0*np.sign(y-d)/d
    h=1.0/d
    return -g,h
def mape(y,d):
    c=d.get_label()
    result=np.sum(np.abs(y-c))/len(c)
    return "mape",result
def mape_ln(y,d):
    c=d.get_label()
    result=np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)
    return "mape",result


###处理time_interval

def AddBaseTimeFeature(df):
    df['start_time']=pd.to_datetime(df['time_interval'].map(lambda x :x[1:20]))
    df['month']=df['start_time'].map(lambda x: x.strftime('%m'))
    df['hour']=df['start_time'].map(lambda x: x.strftime('%H'))
    df['week']=df['start_time'].map(lambda x: x.weekday()+1)
    df['minute']=df['start_time'].map(lambda x: int(x.strftime('%M')))
    df['date']=df['date'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
    #df['date']=pd.to_datetime(df['date'])
    return df

##定义取众数函数

from scipy.stats import mode
def mode_function(df):
    counts=mode(df)
    return counts[0][0]

##读入数据

input_path=r"/Users/evans/Desktop/竞赛/天池智能交通/第二轮/[新-训练集]gy_contest_traveltime_training_data_second.txt"
df=pd.read_csv(input_path,delimiter=';',header=0)
sub=pd.read_csv("/Users/evans/Desktop/竞赛/天池智能交通/第二轮/x_predict_2107.csv",header=0,sep=',')

##合并数据进行统一处理

new_df=pd.concat([df,sub],axis=0) ##axis=0指定了纵向合并
new_df=AddBaseTimeFeature(new_df)

##用pandas的get_dummies函数将week，hour，minute转换为哑变量

week=pd.get_dummies(new_df['week'],prefix='week')
hour=pd.get_dummies(new_df['hour'],prefix='hour')
minute=pd.get_dummies(new_df['minute'],prefix='minute')

new_df=pd.concat([new_df,week,hour,minute],axis=1)


##分组计算一些统计特征作为模型变量
#需预测区间为8点至9点，为简便起见，这里只对8点至9点时间段构造特征，且只考虑6点到8点的相关影响，从总体中抽取时间段在6点到9点的数据，减少后期数据连接的计算量。

data=new_df[new_df['hour'].isin['06','07','08']]

#加入道路信息和pagerank值
link_info=pd.read_table(r'./gy_contest_link_info.txt',header=0,delimiter=';')
page_rank=pd.read_table(r'./page_rank.txt',header=0,sep=',') #page_rank是之前计算好的

data=pd.merge(data,page_rank,on='linkID',how='left')
data=pd.merge(data,link_info,on='linkID',how='left')


H_8=new_df[new_df['hour']=='08']



#1.计算同期历史水平均值和最大值

group_H_M=H_8.groupby(['linkID','hour','minute'])['travel_time'].agg([('history_mean',np.mean),('history_high',np.max)]).reset_index()

#2.计算同期历史水平均值和最大值（考虑星期影响）
group_W_H_m=H_8.groupby(['linkID','week','hour','minute'])['travel_time'].agg([('sameWeek_mean',np.mean),('sameWeek_high',np.max)])\
    .reset_index()

H_8=pd.merge(H_8,group_H_M,on=['linkID','hour','minute'],how='left')
H_8=pd.merge(H_8,group_W_H_m,on=['linkID','week','hour','minute'],how='left')

#3.通过6点-8点时间段的travel_time构造变量

#agg_variables=[('mean_%d' % (j),np.mean), ('median_%d' % (j),np.median),('mode_%d' % (j),mode_function),('std_%d' % (j),np.std),('max_%d' % (j), np.max)]

#训练集使用4月份数据
def compute_vars(df, month):
    for i in ['06','07']:
        for j in range(18,68,10):
            tmp=data[(data['month']==month) & (data['hour']==i) & (data['minute']>=j)]
            tmp=tmp.groupby(['linkID','date'])['travel_time'].agg([('mean_%d' % (j),np.mean), ('median_%d' % (j),np.median),('mode_%d' % (j),mode_function),('std_%d' % (j),np.std),('max_%d' % (j), np.max)]
            ).reset_index()
            df=pd.merge(df,tmp,on=['linkID','date'],how='left')
    return df

train=data[(data['month']=='04') & (data['hour']=='08')]

for i in ['06','07']:
    for j in np.arange[18,58,10]:
        tmp=data[(data['month']=='04') & (data['hour']==i) & (data['minute']>=j)]
        tmp=tmp.groupby(['linkID','date'])['travel_time'].agg(agg_variables).reset_index()
        train=pd.merge(train,tmp,on=['linkID','date'])

##测试集使用5月份数据

test=[(data['month']=='05') & (data['hour']=='08')]


for i in ['06','07']:
    for j in np.arange[18,58,10]:
        tmp=data[(data['month']=='05') & (data['hour']==i) & (data['minute']>=j)]
        tmp=tmp.groupby(['linkID','date'])['travel_time'].agg(agg_variables).reset_index()
        test=pd.merge(test,tmp,on=['linkID','date'])

##预测集-6月份数据

pred=data[(data['month']=='06') & (data['hour']=='08')]

for i in ['06','07']:
    for j in np.arange[18,58,10]:
        tmp=data[(data['month']=='06') & (data['hour']==i) & (data['minute']>=j)]
        tmp=tmp.groupby(['linkID','date'])['travel_time'].agg(agg_variables).reset_index()
        pred=pd.merge(pred,tmp,on=['linkID','date'])



#####以上计算函数版本###
def compute_vars(df, month):
    for i in ['06','07']:
        for j in range(10,60,10):
            tmp=data[(data['month']==month) & (data['hour']==i) & (data['minute']>=j)]
            tmp=tmp.groupby(['linkID','date'])['travel_time'].agg([('mean_%d' % (j),np.mean), ('median_%d' % (j),np.median),('mode_%d' % (j),mode_function),('std_%d' % (j),np.std),('max_%d' % (j), np.max)]
            ).reset_index()
            df=pd.merge(df,tmp,on=['linkID','date'],how='left')
    return df
def get_last_month_value(df,month):
    tmp=data[(data['month']==month) & (data['hour']=='08')]
    tmp1=tmp.groupby(['linkID','minute'])['travel_time'].agg([('mean',np.mean),('max',np.max),('min',np.min)]).reset_index()
    df=pd.merge(df,tmp1,on=['linkID','minute'],how='left')
    tmp2=tmp.groupby(['linkID','week','minute'])['travel_time'].agg([('mean_sw',np.mean),('max_sw',np.max),('min_sw',np.min)]).reset_index()
    df=pd.merge(df,tmp2,on=['linkID','week','minute'],how='left')
    return df
def get_lag1_lag2(df,month,hour1='07',hour2='06'):
    tmp1=data[(data['month']==month) & (data['hour']==hour1)]
    tmp2=data[(data['month']==month) & (data['hour']==hour2)]
    tmp1=tmp1.loc[:,['linkID','start_time','month','hour','minute','date','travel_time']]
    tmp2=tmp2.loc[:,['linkID','start_time','month','hour','minute','date','travel_time']]
    tmp1.rename(columns={'travel_time':'lag1'},inplace=True)
    tmp2.rename(columns={'travel_time':'lag2'},inplace=True)
    tmp1['start_time']=tmp1['start_time'].map(lambda x: x+timedelta(hours=1))
    tmp1['hour']=tmp1['start_time'].map(lambda x: x.strftime('%H'))
    tmp2['start_time']=tmp2['start_time'].map(lambda x: x+timedelta(hours=2))
    tmp2['hour']=tmp2['start_time'].map(lambda x: x.strftime('%H'))
    df=pd.merge(df,tmp1,on=['linkID','start_time','date','hour','minute','month'],how='left')
    df=pd.merge(df,tmp2,on=['linkID','start_time','date','hour','minute','month'],how='left')
    return df




###模型训练

train_label=np.log1p(train.pop('travel_time'))
test_label=np.log1p(test.pop('travel_time'))
train_2=train.drop(['hour','month','start_time','minute','date','week'],axis=1)
test_2=test.drop(['hour','month','start_time','minute','date','week'],axis=1)


import xgboost as xgb
xlf=xgb.XGBRegressor(max_depth=11,
                     learning_rate=0.01,
                     n_estimators=301,
                     silent=True,
                     objective=mape_object,
                       gamma=0,
                       min_child_weight=5,
                       max_delta_step=0,
                       subsample=0.8,
                       colsample_bytree=0.8,
                       colsample_bylevel=1,
                       reg_alpha=1e0,
                       reg_lambda=0,
                       scale_pos_weight=1,
                       seed=9,
                       missing=None)


xlf.fit(train_2.values,train_label.values,eval_metric=mape_ln,verbose=True,eval_set=[(test_2.values,test_label.values)],early_stopping_rounds=2)
print(xlf.get_params())

pred.pop('travel_time')
pred_2=pred.drop(['hour','month','start_time','minute','date','week'],axis=1)

predict_time=xlf.predict(pred.values)
travel_time=pd.DataFrame({'travel_time':list(predict_time)})

pred_3=pd.concat([pred,travel_time],axis=1)

submit=pred_3.loc[:,'linkID','date','start_time','travel_time']

def get_time_interval(x):
    y=x+timedelta(minutes=2)
    z='['+str(x)+','+str(y)+')'
    return z

submit['travel_time']=np.round(np.expm1(submit['travel_time']),6)
submit['time_interval']=submit['start_time'].map(lambda x: get_time_interval(x))

submit.loc[:,['linkID','date','start_time','travel_time']].to_csv("./submit.txt",sep="#",header=False,index=False)


########加入同期滞后1期和2期的数据--------


def get_lag1_lag2(df,month,hour1='07',hour2='06'):
    tmp1=data[(data['month']==month) & (data['hour']==hour1)]
    tmp2=data[(data['month']==month) & (data['hour']==hour2)]
    tmp1=tmp1.loc[:,['linkID','start_time','month','hour','minute','date','travel_time']]
    tmp2=tmp2.loc[:,['linkID','start_time','month','hour','minute','date','travel_time']]
    tmp1.rename(columns={'travel_time':'lag1'},inplace=True)
    tmp2.rename(columns={'travel_time':'lag2'},inplace=True)
    tmp1['start_time']=tmp1['start_time'].map(lambda x: x+timedelta(hours=1))
    tmp1['hour']=tmp1['start_time'].map(lambda x: x.strftime('%H'))
    tmp2['start_time']=tmp2['start_time'].map(lambda x: x+timedelta(hours=2))
    tmp2['hour']=tmp2['start_time'].map(lambda x: x.strftime('%H'))
    df=pd.merge(df,tmp1,on=['linkID','start_time','date','hour','minute','month'],how='left')
    df=pd.merge(df,tmp2,on=['linkID','start_time','date','hour','minute','month'],how='left')
    return df





####------------------预测集构造方法-------------------#####

from datetime import datetime
from datetime import timedelta

def dateRange(beginDate, endDate):
    datetimes = []
    dt = datetime.datetime.strptime(beginDate, "%Y-%m-%d %H:%M:%S")
    time = beginDate[:]
    while time <= endDate:
        datetimes.append(time)
        dt = dt + datetime.timedelta(minutes=2)
        time = dt.strftime("%Y-%m-%d %H:%M:%S")
    return datetimes

days=[]
for i in range(1,31):
    if i <10:
        z='0'+str(i)
    else:
        z=str(i)
    days.append(i)

datetimes=[]
for i in days:
    datetimes.append(dateRange("2017-06-%s 08:00:00" %i,"2017-06-%s 08:58:00" %i))

time_frame = pd.DataFrame({'start_time': datetimes})
link_ID=list(set(link_info['link_ID']))



# 得到笛卡尔积
import itertools

link_time=[]
for x in itertools.product(link_ID,datetimes):
    link_time.append(x)

June_data=pd.DataFrame({'link_time':link_time})

def get_time_interval(x):
    y=x+timedelta(minutes=2)
    z="["+str(x)+","+str(y)+")"
    return z

def get_sub_data(df):
    df['linkID']=df['link_time'].map(lambda x: x[0])
    df['start_time']=df['link_time'].map(lambda x:datetime.datetime.strptime(x[1],"%Y-%m-%d %H:%M:%S"))
    df['date']=df['start_time'].map(lambda x: x.date())
    df['time_interval']=df['start_time'].map(lambda x: get_time_interval(x))
    df=df.loc[:,['linkID','date','time_interval']]
    return df

June_data_demo.to_csv('./June_data_demo.csv', index=False, header=False,encoding='utf_8')