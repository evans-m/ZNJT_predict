#conding=UTF-8
##
##Created on 2017年08月08日

## @author: evans

##用历史均值作为预测值
import pandas as pd
import pickle
import numpy as np
import datetime
df=pd.read_csv(r"/Users/evans/Desktop/竞赛/天池智能交通/cleand_df_withDateTime.csv",header=0)
x_pred=pd.read_csv("/Users/evans/Desktop/竞赛/天池智能交通/x_predict.csv",header=0)

new_df=df[df['hour']==8]
variables=['link_ID','pr_full','weekday', 'hour', 'minute','month','startTime','travel_time']
new_df=new_df.loc[:,variables]
##分组方式1
groupd=new_df['travel_time'].groupby([new_df['link_ID'],new_df['minute']])
agg1=groupd.aggregate(np.mean).reset_index()
s1=open("/Users/evans/Desktop/竞赛/天池智能交通/agg1.pkl",'wb')
pickle.dump(agg1,s1)
##分组方式2：考虑星期数的影响
gb=new_df.groupby([new_df['link_ID'],new_df['minute'],new_df['weekday']])
agg2=gb.aggregate(np.mean).reset_index()
s2=open("/Users/evans/Desktop/竞赛/天池智能交通/agg2.pkl",'wb')
pickle.dump(agg2,s2)
agg2.to_csv(r"/Users/evans/Desktop/竞赛/天池智能交通/agg2.csv")
###
def get_value_by_week(i):
    clue=(agg2['link_ID']==x_pred['link_ID'][i]) & (agg2['minute']==x_pred['minute'][i])\
         & (agg2['weekday']==x_pred['weekday'][i])
    return agg2['travel_time'][clue].values

def get_value_by_minute(i):
    n = x_pred['travel_time'][i]
    if n == []:
        clue = (agg1['link_ID'] == x_pred['link_ID'][i]) & (agg1['minute'] == x_pred['minute'][i])
        return agg1['travel_time'][clue].values
    else:
        return x_pred['travel_time'][i]
##预测数据集是已经做好的，代码将在另一个py文件中给出
import datetime
x_pred['pred_travelTime']=[get_value_by_week(x) for x in range(len(x_pred))]
x_pred['pred_travel_time']=[get_value_by_minute(x) for x in range(len(x_pred))]
submit1=x_pred.loc[:,['link_ID','start_time','travel_time','pred_travel_time']]
submit1['start_time']=[datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S') for x in submit1['start_time']]
submit1['date']=[x.strftime('%Y-%m-%d') for x in submit1['start_time']]
submit1['end_time']=[x+datetime.timedelta(minutes=2) for x in submit1['start_time']]
submit1['time_interval']=['('+str(submit1['start_time'][i])+','+str(submit1['end_time'][i])+']'\
                          for i in range(len(submit1))]
def travel_time(x):
    if x==[]:
        return 'missing'
    else:
        return x[0]
submit1['travel_time']=[travel_time(x) for x in submit1['travel_time']]
submit1=submit1.loc[:,['link_ID','time_interval','travel_time']]
submit1=submit1[submit1['travel_time'] != 'missing']
submit1.to_csv(r"/Users/evans/Desktop/竞赛/天池智能交通/submit1_evans.csv")
submit1.to_csv(r"/Users/evans/Desktop/竞赛/天池智能交通/submit1_evans.txt",index=False,header=False,sep='#')