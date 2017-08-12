#conding=UTF-8
'''
Created on 2017年08月04日

 @author: evans
'''

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

link_info=pd.read_table(r'/Users/evans/Desktop/竞赛/天池智能交通/gy_contest_link_info.txt',header=0,delimiter=';')
link_top=pd.read_table(r'/Users/evans/Desktop/竞赛/天池智能交通/gy_contest_link_top(20170715更新).txt',header=0,delimiter=';')
traveltime=pd.read_table(r'/Users/evans/Desktop/竞赛/天池智能交通/gy_contest_link_traveltime_training_data.txt',header=0,delimiter=';')
sample_data=pd.read_table(r'/Users/evans/Desktop/竞赛/天池智能交通/sample_data_10.txt',header=None,delimiter='#')
sample_data=pd.DataFrame({'link_ID':sample_data[0],'date':sample_data[1],'time_interval':sample_data[2],\
                          'travel_time':sample_data[3]})


link_relation=link_top.fillna("missing")
link_relation['ins']=[x.split("#") for x in link_relation['in_links']]
link_relation['outs']=[x.split("#") for x in link_relation['out_links']]
#构建图模型的边
in_list=[]
for i in range(len(link_relation)):
    for j in link_relation['ins'][i]:
        if j !='missing':
            in_list.append((j,str(link_relation.loc[i,'link_ID'])))
out_list=[]
for i in range(len(link_relation)):
    for j in link_relation['outs'][i]:
        if j != 'missing':
            out_list.append((str(link_relation.loc[i,'link_ID']),j))
#构建图模型
G=nx.DiGraph()
G.add_edges_from(in_list)
G.add_edges_from(out_list)

layout=nx.spring_layout(G)
fig=plt.figure(figsize=(12,8))
nx.draw(G,pos=layout,node_color='y',with_labels=False,node_size=30,style='dashed')
plt.show()

fig.savefig(r'/Users/evans/Desktop/竞赛/天池智能交通/road_net_full.png',dpi=fig.dpi)
#计算pagerank值
pagerank=nx.pagerank(G,alpha=0.85)
ID=[]
for k in pagerank_full.keys():
    ID.append(k)
ID_pr=pd.DataFrame({'ID':ID})
ID_pr['pr_full']=[pagerank.get(k) for k in ID_pr['ID']]
#union数据集
from pandas import merge
new_df=merge(traveltime,link_info,how='left',on='link_ID')
new_df['ID']=[str(x) for x in new_df['link_ID']]
cleand_df=merge(new_df,ID_pr,how='left',on='ID')
cleand_df.to_csv(r'/Users/evans/Desktop/竞赛/天池智能交通/df_with_pagerank.txt')
ID_pr.to_csv(r'/Users/evans/Desktop/竞赛/天池智能交通/page_rank.txt')
cleand_df.to_csv(r'/Users/evans/Desktop/竞赛/天池智能交通/df_with_pagerank.csv')
ID_pr.to_csv(r'/Users/evans/Desktop/竞赛/天池智能交通/page_rank.csv')

cleand_df=cleand_df.dropna()
cleand_df['interval_split']=[x.split(",") for x in cleand_df['time_interval']]
cleand_df['start_time']=[x[0].strip("[") for x in cleand_df['interval_split']]
cleand_df['end_time']=[x[1].strip(")") for x in cleand_df['interval_split']]
import datetime
cleand_df['startTime']=[datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S') for x in cleand_df['start_time']]
cleand_df['endTime']=[datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S') for x in cleand_df['end_time']]

cleand_df['weekday']=[x.weekday() for x in cleand_df['date']]
cleand_df['hour']=[x.strftime('%H') for x in cleand_df['startTime']]
cleand_df['minute']=[x.strftime('%M') for x in cleand_df['startTime']]
cleand_df['month']=[x.strftime('%m') for x in cleand_df['date']]
#cleand_df['day']=[x.strftime('%d') for x in cleand_df['date']]
#cleand_df.columns
variables=['length', 'width','pr_full','weekday', 'hour', 'minute',
       'month']
target=['travel_time']
cleand_df.to_csv(r'/Users/evans/Desktop/竞赛/天池智能交通/cleand_df_withDateTime.csv')
df.to_pickle('clean_df')
#划分训练集与测试集
from sklearn import cross_validation
x=cleand_df.loc[:,variables]
y=cleand_df.loc[:,target]
X_train,X_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.4,random_state=0)
#拟合SVR模型
from sklearn import svm
clf1=svm.SVR()
clf1.fit(X_train,y_train)
<<<<<<< Updated upstream:zhjt_predict.py
=======
##随机抽取样本容量位20000的小样本
sample1=df.sample(n=20000,random_state=12345)

sample1_x=sample1.loc[:,variables]

sample1_y=sample1.lon[:,target]

sample1_clf=svm.SVR()
sample1_clf.fit(sample1_X_train,sample1_y_train)
sample_y_predictd=sample1_clf.predict(sample1_X_test)
##标准化，按训练集进行标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(sample1_X_train)
X_train=scaler.transform(sample1_X_train)
X_test=scaler.transform(sample1_X_test)
##多层感知机模型
from sklearn.neural_network import MLPRegressor
clf2=MLPRegressor()
##保存模型
from sklearn.externals import joblib
joblib.dump(clf,"SVR_1.pkl")
>>>>>>> Stashed changes:data_prepare.py
