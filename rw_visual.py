import pandas as pd
import re
import datetime
import numpy as np
import math
import random
import glob
import numpy as np

data=pd.read_excel('G:\SQL_CJ\wuhan\estate.xls')
group_estate=pd.pivot_table(data,index=['板块'],values=['小区名称'],aggfunc=lambda x:len(x))
s=group_estate.reset_index(drop=True)
b=[]
for i in group_estate.index:
    b.append(i)
b=pd.DataFrame(b,columns=['板块'])
group_estate=pd.concat([b,s],axis=1)
data=pd.merge(data,group_estate,how='left',on='板块')
data=data.ix[:,['小区名称_x','地址','坐标','行政区','板块','建成年份','小区名称_y','参考均价']]
data.columns=['小区名称','地址','坐标','行政区','板块','建成年份','板块小区数量','参考均价']
f=lambda x:re.sub("\D", "", x)
data['建成年份']=data['建成年份'].map(f)
price=data['参考均价'].str.split('元',expand=True)
data['参考均价']=price[0]
data['参考均价']=data['参考均价'].astype(float)
f0=lambda x:0 if x=='' else x
data['建成年份']=data['建成年份'].map(f0)
df_t=[]
for i in data['建成年份']:
    i=int(i)
    df_t.append(i)
df_t=pd.DataFrame(df_t)
df_t=datetime.datetime.now().year-df_t
data['房龄']=df_t
fot=lambda x:None if x==datetime.datetime.now().year else x
data['房龄']=data['房龄'].map(fot)
fot3=lambda x:None if x<0 else x
data['房龄']=data['房龄'].map(fot3)

#成交文件合并
files=glob.glob('G:\SQL_CJ\wuhan\deal\*.csv')
df = pd.DataFrame(columns=['城市','小区名称', '户型', '建筑面积', '朝向','装修', '有无电梯','楼层','建成年份', '标签1','标签2','挂牌总价1','挂牌总价2','成交时间','成交总价', '成交单价', '成交周期1','成交周期2','页面标题','采集时间'])
for csv in files:
    df = pd.merge(df,pd.read_csv(csv),how='outer')
data1 = pd.DataFrame(df,columns=['城市','小区名称', '户型', '建筑面积', '朝向','装修', '有无电梯','楼层','建成年份', '标签1','标签2','挂牌总价1','挂牌总价2','成交时间','成交总价', '成交单价', '成交周期1','成交周期2','页面标题','采集时间'])
data1['成交周期天']=data1['成交周期2']
data1=data1.drop_duplicates(subset=['小区名称','户型','建筑面积','装修','楼层','建成年份','成交时间','成交周期天'])


#成交数据处理
s=data1['建筑面积'].str.split('平',expand=True)
data1['建筑面积']=s[0]
data1['建筑面积'] = data1['建筑面积'].astype(float)
p=data1['成交单价'].str.split('元',expand=True)
data1['成交单价']=p[0]
price=data1['挂牌总价2'].str.split('万',expand=True)
data1['挂牌总价2']=price[0]
data1['挂牌总价2'] = data1['挂牌总价2'].astype(float)
data1['成交单价'][data1['成交时间']=='近30天内成交']=data1['挂牌总价2'][data1['成交时间']=='近30天内成交']/data1['建筑面积'][data1['成交时间']=='近30天内成交']*10000
t=data1['成交周期天'].str.split('天',expand=True)
data1['成交周期天']=t[0]
data1=data1[data1['成交周期天'].notnull()]
fomat=lambda x:x.replace('近30天内成交','2018.10.01')
data1['成交时间']=data1['成交时间'].map(fomat)
times=data1['成交时间'].str.split('.',expand=True)
times.columns=['年','月','日']
fot_y=lambda x:int(x)
data1['成交周期天']=data1['成交周期天'].map(fot_y)
#data1=data1[data1['成交周期天']<365]



times['nian']=times['年'].map(fot_y)
times['yue']=times['月'].map(fot_y)
data1=pd.concat([data1,times],axis=1)
df=pd.merge(data1,data,how='left',on='小区名称')
data1=data1[data1['nian']==datetime.datetime.now().year ]
data1=data1[data1['yue']>=datetime.datetime.now().month-6]
ct=pd.pivot_table(data1,index=['小区名称'],values=['成交周期天'],aggfunc=lambda x:len(x))

ct_0=ct.reset_index(drop=True)
b_t=[]
for i in ct.index:
    b_t.append(i)
b_t=pd.DataFrame(b_t,columns=['小区名称'])
ct_estate=pd.concat([b_t,ct_0],axis=1)
ct_estate.columns=['小区名称','案例个数']
data1=pd.merge(data1,ct_estate,how='left',on='小区名称')
data1=data1[data1['案例个数']>=3]
group_time=pd.pivot_table(data1,index=['小区名称'],values=['成交周期天'],aggfunc=np.median)
s1=group_time.reset_index(drop=True)
b1=[]
for i in group_time.index:
    b1.append(i)
b1=pd.DataFrame(b1,columns=['小区名称'])
group_time=pd.concat([b1,s1],axis=1)
data2=pd.merge(data,group_time,how='left',on='小区名称')
data1=data1[data1['成交单价'].notnull()]
data1=data1.reset_index(drop=True)
data1['成交单价'] = data1['成交单价'].astype(int)
group_price=pd.pivot_table(data1,index=['小区名称'],values=['成交单价'],aggfunc=np.median)
s2=group_price.reset_index(drop=True)
b2=[]
for i in group_price.index:
    b2.append(i)
b2=pd.DataFrame(b2,columns=['小区名称'])
group_price=pd.concat([b2,s2],axis=1)
data3=pd.merge(data2,group_price,how='left',on='小区名称')
#data3=pd.merge(data3,cj_b,how='left',on='板块')
data3['武汉市成交周期（天）']=data1['成交周期天'].mean()
cj_pb=data3['成交单价'].groupby(data3['板块']).mean()
cj_pb=pd.DataFrame(cj_pb)
cj_pb1=cj_pb.reset_index(drop=True)
a3=[]
for i in cj_pb.index:
    a3.append(i)
a3=pd.DataFrame(a3,columns=['板块'])
cj_pb=pd.concat([a3,cj_pb1],axis=1)
data3=pd.merge(data3,cj_pb,how='left',on='板块')
cj_px=data3['成交单价_x'].groupby(data3['行政区']).mean()
cj_px=pd.DataFrame(cj_px)
cj_px1=cj_px.reset_index(drop=True)
a4=[]
for i in cj_px.index:
    a4.append(i)
a4=pd.DataFrame(a4,columns=['行政区'])
cj_px=pd.concat([a4,cj_px1],axis=1)
data3=pd.merge(data3,cj_px,how='left',on='行政区')
data3.rename(columns={'成交单价_x_x':'成交单价','成交单价_y':'板块成交单价','成交单价_x_y':'行政区成交单价'},inplace = True)
data3['武汉市成交单价']=data3['成交单价'].median()

#挂牌
files=glob.glob('G:\SQL_CJ\wuhan\sale\*.csv')
df_g = pd.DataFrame(columns=['城市','板块','小区名称','房型','建筑面积','装修', '有无电梯', '楼层', '挂牌总价', '挂牌单价',  '关注', '带看','发布', '标题','标签1','标签2', '标签3', '采集时间','页面标题'])
for csv in files:
    df_g = pd.merge(df_g,pd.read_csv(csv),how='outer')
data4= pd.DataFrame(df_g,columns=['城市','板块','小区名称','房型','建筑面积','装修', '有无电梯', '楼层', '挂牌总价', '挂牌单价',  '关注', '带看','发布', '标题','标签1','标签2', '标签3', '采集时间','页面标题'])
data4=data4[data4['发布'].notnull()]
data4.to_csv('G:\SQL_CJ\wuhan\sale.csv',index=None)

g=data4['关注'].str.split('人',expand=True)
data4['关注']=g[0]
d=data4['带看'].str.split('次',expand=True)
data4['带看']=d[0]
data4['发布']=data4['发布'].astype(str)
names_1=data4['发布'].str.split('个',expand=True)
ft=lambda x:x.replace('一年前发布','12')
names_1[0]=names_1[0].map(ft)
names_2=names_1[0].str.split('天',expand=True)
ft1=lambda x:x.replace('刚刚发布','1')
names_2[0]=names_2[0].map(ft1)
ft2=lambda x:int(x)
names_2[0]=names_2[0].map(ft2)
ft3=lambda x:x*30+random.randint(0,30) if x<=12 else x
names_2[0]=names_2[0].map(ft3)
num1=data4['房型'].str.split('室',expand=True)
type=num1[0]
type=pd.DataFrame(type)
type.columns=['户型']
df=pd.concat([data4['小区名称'],data4['关注'],data4['带看'],type,names_2[0]],axis=1)
df1=df.ix[:,['小区名称','户型','关注','带看',0]]
df1.columns=['小区名称','户型','关注量','带看量','上架周期']
df1=df1[df1['户型'].notnull()]
df2=df1.ix[:,1:]
ft4=lambda x:int(x)
df2=df2.applymap(ft4)
df1=pd.concat([df1['小区名称'],df2],axis=1)
df1['关注量']=(df1['关注量']/df1['上架周期'])*30
df1['带看量']=(df1['带看量']/df1['上架周期'])*30
df3=df1[df1['上架周期']<365]
sj_t=df3['上架周期'].groupby(df3['小区名称']).median()
sj_t=pd.DataFrame(sj_t)
sj_t1=sj_t.reset_index(drop=True)
c1=[]
for i in sj_t.index:
    c1.append(i)
c1=pd.DataFrame(c1,columns=['小区名称'])
sj_t=pd.concat([c1,sj_t1],axis=1)
data3=pd.merge(data3,sj_t,how='left',on='小区名称')
data3['武汉平均上架周期（天）']=df1['上架周期'].median()
df1=df1[df1['上架周期']<90]
group_p1=pd.pivot_table(df1,index=['小区名称'],values=['关注量','带看量'],aggfunc=np.median)
#小区整体关注指数
group_p1['关注指数']=group_p1['关注量']+5.0*group_p1['带看量']
#小区按户型关注指数
group_p2=pd.pivot_table(df1,index=['小区名称','户型'],values=['关注量','带看量'],aggfunc=np.median)
group_p2['关注指数']=0.5*group_p2['关注量']+5.0*group_p2['带看量']
group_p2=group_p2['关注指数'].unstack(1)
group_p2=group_p2.fillna(0)

group_p2['4室以上']=group_p2[5]+group_p2[6]+group_p2[7]
group_p2['指数']=group_p1['关注指数']
group_p2=group_p2.ix[:,['指数',1,2,3,4,'4室以上']]
group_p2.columns=['小区关注指数','1室关注指数','2室关注指数','3室关注指数','4室关注指数','4室以上关注指数']
s2=group_p2.reset_index(drop=True)
b2=[]
for i in group_p2.index:
    b2.append(i)
b2=pd.DataFrame(b2,columns=['小区名称'])
group_p2=pd.concat([b2,s2],axis=1)
data_set=pd.merge(data3,group_p2,how='left',on='小区名称')

#data_set=pd.merge(data3,group_p2,how='left',on='小区名称')
#data_set['小区关注指数排名']=data_set['小区关注指数'].groupby(data_set['板块']).rank(ascending=False)
data_set['基准日']=datetime.datetime.today()


data_set.rename(columns={'上架周期':'上架周期（天）'},inplace = True)
data_set_notnull=data_set[(data_set['成交周期天'].notnull()) & (data_set['上架周期（天）'].notnull()) & (data_set['小区关注指数'].notnull())]
data_set_notnull['指标缺失']='否'

data_set_null=data_set[(data_set['成交周期天'].isnull()) | (data_set['上架周期（天）'].isnull()) | (data_set['小区关注指数'].isnull())]
data_set_null['指标缺失']='是'
data_set_null=data_set_null.reset_index(drop=True)
for i in data_set_null.index:
    if data_set_null.ix[[i], ['成交周期天']].isnull:
        t = data_set_null.ix[[i], ['房龄']]
        f = lambda x: int(x) if x == x else -3
        t = t.applymap(f)
        t = np.array(t)
        t = int(t)
        p =data_set_null.ix[[i], ['参考均价']]
        f1 = lambda x: int(x) if x == x else 0
        p = p.applymap(f1)
        p = np.array(p)
        p = int(p)
        data_set_null.ix[[i], ['成交周期天']] = data_set_notnull['成交周期天'][
            (abs(data_set_notnull['房龄'] - t) <= 2) & (abs(data_set_notnull['参考均价'] / p - 1) <= 0.15)].mean()
    if data_set_null.ix[[i], ['上架周期（天）']].isnull:
        t = data_set_null.ix[[i], ['房龄']]
        f = lambda x: int(x) if x == x else -3
        t = t.applymap(f)
        t = np.array(t)
        t = int(t)
        p =data_set_null.ix[[i], ['参考均价']]
        f1 = lambda x: int(x) if x == x else 0
        p = p.applymap(f1)
        p = np.array(p)
        p = int(p)
        data_set_null.ix[[i], ['上架周期（天）']] = data_set_notnull['上架周期（天）'][
                (abs(data_set_notnull['房龄'] - t) <= 2) & (abs(data_set_notnull['参考均价'] / p - 1) <= 0.15)].mean()
    if data_set_null.ix[[i], ['小区关注指数']].isnull:
        t = data_set_null.ix[[i], ['房龄']]
        f = lambda x: int(x) if x == x else -3
        t = t.applymap(f)
        t = np.array(t)
        t = int(t)
        p =data_set_null.ix[[i], ['参考均价']]
        f1 = lambda x: int(x) if x == x else 0
        p = p.applymap(f1)
        p = np.array(p)
        p = int(p)
        data_set_null.ix[[i], ['小区关注指数']] = data_set_notnull['小区关注指数'][
                (abs(data_set_notnull['房龄'] - t) <= 2) & (abs(data_set_notnull['参考均价'] / p - 1) <= 0.15)].mean()
data_set=pd.concat([data_set_null,data_set_notnull],axis=0)
data_set_notnull=data_set[(data_set['成交周期天'].notnull()) & (data_set['上架周期（天）'].notnull()) & (data_set['小区关注指数'].notnull())]
data_set_null=data_set[(data_set['成交周期天'].isnull()) | (data_set['上架周期（天）'].isnull()) | (data_set['小区关注指数'].isnull())]
data_set_null=data_set_null.reset_index(drop=True)
for i in data_set_null.index:
    if data_set_null.ix[[i], ['成交周期天']].isnull:
        t = data_set_null.ix[[i], ['房龄']]
        f = lambda x: int(x) if x == x else -3
        t = t.applymap(f)
        t = np.array(t)
        t = int(t)
        p =data_set_null.ix[[i], ['参考均价']]
        f1 = lambda x: int(x) if x == x else 0
        p = p.applymap(f1)
        p = np.array(p)
        p = int(p)
        data_set_null.ix[[i], ['成交周期天']] = data_set_notnull['成交周期天'][
            (abs(data_set_notnull['房龄'] - t) <= 2) & (abs(data_set_notnull['参考均价'] / p - 1) <= 0.15)].mean()
    if data_set_null.ix[[i], ['上架周期（天）']].isnull:
        t = data_set_null.ix[[i], ['房龄']]
        f = lambda x: int(x) if x == x else -3
        t = t.applymap(f)
        t = np.array(t)
        t = int(t)
        p =data_set_null.ix[[i], ['参考均价']]
        f1 = lambda x: int(x) if x == x else 0
        p = p.applymap(f1)
        p = np.array(p)
        p = int(p)
        data_set_null.ix[[i], ['上架周期（天）']] = data_set_notnull['上架周期（天）'][
                (abs(data_set_notnull['房龄'] - t) <= 2) & (abs(data_set_notnull['参考均价'] / p - 1) <= 0.15)].mean()
    if data_set_null.ix[[i], ['小区关注指数']].isnull:
        t = data_set_null.ix[[i], ['房龄']]
        f = lambda x: int(x) if x == x else -3
        t = t.applymap(f)
        t = np.array(t)
        t = int(t)
        p =data_set_null.ix[[i], ['参考均价']]
        f1 = lambda x: int(x) if x == x else 0
        p = p.applymap(f1)
        p = np.array(p)
        p = int(p)
        data_set_null.ix[[i], ['小区关注指数']] = data_set_notnull['小区关注指数'][
                (abs(data_set_notnull['房龄'] - t) <= 2) & (abs(data_set_notnull['参考均价'] / p - 1) <= 0.15)].mean()
qw=data_set['成交周期天'].groupby(data_set['板块']).mean()
cw=pd.DataFrame(qw)
cw1=cw.reset_index(drop=True)
n2=[]
for i in cw.index:
    n2.append(i)
n2=pd.DataFrame(n2,columns=['板块'])
cw=pd.concat([n2,cw1],axis=1)
data_set=pd.merge(data_set,cw,how='left',on='板块')
#data_set.rename(columns={,'成交周期（天）_x':'成交周期（天）'},inplace = True)
qt=data_set['成交周期天_x'].groupby(data_set['行政区']).mean()
aw=pd.DataFrame(qt)
aw1=aw.reset_index(drop=True)
n3=[]
for i in aw.index:
    n3.append(i)
n3=pd.DataFrame(n3,columns=['行政区'])
aw=pd.concat([n3,aw1],axis=1)
data_set=pd.merge(data_set,aw,how='left',on='行政区')
data_set.rename(columns={'成交周期天_x_x':'成交周期（天）','成交周期天_y':'板块成交周期（天）','成交周期天_x_y':'行政区成交周期（天）'},inplace = True)
bw=data_set['上架周期（天）'].groupby(data_set['板块']).mean()

bw=pd.DataFrame(bw)
bw1=bw.reset_index(drop=True)
n4=[]
for i in bw.index:
    n4.append(i)
n4=pd.DataFrame(n4,columns=['板块'])
bw=pd.concat([n4,bw1],axis=1)
data_set=pd.merge(data_set,bw,how='left',on='板块')
#data_set.rename(columns={'上架周期（天）_y':'板块上架周期（天）','上架周期（天）_x':'上架周期（天）'},inplace = True)
dw=data_set['上架周期（天）_x'].groupby(data_set['行政区']).mean()
dw=pd.DataFrame(dw)
dw1=dw.reset_index(drop=True)
n5=[]
for i in dw.index:
    n5.append(i)
n5=pd.DataFrame(n5,columns=['行政区'])
dw=pd.concat([n5,dw1],axis=1)
data_set=pd.merge(data_set,dw,how='left',on='行政区')
data_set.rename(columns={'上架周期（天）_x_x':'上架周期（天）','上架周期（天）_y':'板块上架周期（天）','上架周期（天）_x_y':'行政区上架周期（天）'},inplace = True)

ew=data_set['小区关注指数'].groupby(data_set['板块']).mean()
ew=pd.DataFrame(ew)
ew1=ew.reset_index(drop=True)
n6=[]
for i in ew.index:
    n6.append(i)
n6=pd.DataFrame(n6,columns=['板块'])
ew=pd.concat([n6,ew1],axis=1)
data_set=pd.merge(data_set,ew,how='left',on='板块')
#data_set.rename(columns={'小区关注指数_y':'板块关注指数','小区关注指数_x':'小区关注指数'},inplace = True)
fw=data_set['小区关注指数_x'].groupby(data_set['行政区']).mean()
fw=pd.DataFrame(fw)
fw1=fw.reset_index(drop=True)
n7=[]
for i in fw.index:
    n7.append(i)
n7=pd.DataFrame(n7,columns=['行政区'])
fw=pd.concat([n7,fw1],axis=1)
data_set=pd.merge(data_set,fw,how='left',on='行政区')
data_set.rename(columns={'小区关注指数_x_x':'小区关注指数','小区关注指数_y':'板块关注指数','小区关注指数_x_y':'行政区关注指数'},inplace = True)
data_set['武汉市关注指数']=data_set['小区关注指数'].mean()
data_set['成交周期（天）'][data_set['指标缺失']=='是']=data_set['成交周期（天）'][data_set['指标缺失']=='是']*0.7+data_set['板块成交周期（天）'][data_set['指标缺失']=='是']*0.3
data_set['上架周期（天）'][data_set['指标缺失']=='是']=data_set['上架周期（天）'][data_set['指标缺失']=='是']*0.7+data_set['板块上架周期（天）'][data_set['指标缺失']=='是']*0.3
data_set['小区关注指数'][data_set['指标缺失']=='是']=data_set['小区关注指数'][data_set['指标缺失']=='是']*0.7+data_set['板块关注指数'][data_set['指标缺失']=='是']*0.3
data_set['处置周期']=0.55*data_set['成交周期（天）']+0.3*data_set['上架周期（天）']+0.15*(100/(data_set['小区关注指数'].map(lambda x:math.log(1.5+x))))
data_set=data_set.ix[:,['小区名称','地址','行政区','板块','房龄','参考均价','成交单价','板块成交单价','行政区成交单价','武汉市成交单价','成交周期（天）','板块成交周期（天）','行政区成交周期（天）','武汉市成交周期（天）','上架周期（天）','板块上架周期（天）','行政区上架周期（天）','武汉平均上架周期（天）','小区关注指数','板块关注指数','行政区关注指数','武汉市关注指数','处置周期','基准日','指标缺失']]
data_set=data_set[data_set['小区名称'].notnull()]
data_set.to_excel('G:\SQL_CJ\wuhanczzq.xlsx',index=None)
