#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing neccessary libraries
# 导入所需要的包

# 数据处理
import pandas as pd
import numpy as np

# 对数据进行绘制图形分析
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# 处理时间序列信息
import datetime
from datetime import date
import os


# LabelEncoder 编码，特征工程处理
# 使用 sklearn 集成的
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

# 使用模型 XGBClassifier
# 采用 xgboost 集成的
from xgboost.sklearn import XGBClassifier

# 利用 sklearn 集成的一些评测指标和模型选择
import sklearn as sk
from sklearn.metrics import *
from sklearn.model_selection import *

# 用于高效存储数据，读写高效，内存性能
import pickle 

# 警告过滤器
import warnings
# # 警告过滤器用于控制警告的行为，如忽略匹配的警告
warnings.filterwarnings('ignore')


# In[2]:


# Loading the Data
# 加载数据集

# 设置数据集搜索路径
dataset_path = './../dataset/'

# 加载 训练集 和 测试集
df_train = pd.read_csv(dataset_path + 'train_users_2.csv.zip')
df_test = pd.read_csv(dataset_path + 'test_users.csv.zip')

df_train.shape, df_test.shape


# In[3]:


df_train.head()


# In[4]:


df_test.tail()


# In[5]:


# 数据简要分析 summary

# 1、train 训练集数据 比 test 测试集数据 多一个特征 —— country_destination
# 2、country_destination 即就是需要预测的目标变量
# 3、对数据集的探索着重分析 train 训练集数据，test 测试集数据类似

# 查看数据集信息概述
df_train.info()


# In[6]:


# 从数据集的概述中可知：

# 1、trian 数据集包含 213451 行数据，16 个特征
# 2、每个特征的 数据类型 和 非空数量
# 3、date_first_booking 空值较多，与总数据量差一个量级，在特征提取时可以考虑删除
# 4、age 有空值， first_affiliate_tracked 有空值


# In[7]:


# 特征分析工程

# 0 id  用户ID，这个特征没有分析的必要

# 1 date_account_created 用户注册时间，这个特征值得分析

# 查看 date_account_created 特征数据
df_train.date_account_created.head()


# In[8]:


# 对 date_account_created 特征数据进行统计计数
df_train.date_account_created.value_counts().head()


# In[9]:


# 查看 date_account_created 特征的描述信息
df_train.date_account_created.describe()


# In[10]:


# 从以上对 date_account_created 特征的描述信息可知

# 1、该特征 date_account_created 的总计数为 213451
# 2、该特征 date_account_created 不同天数为 1634
# 3、该特征 date_account_created 频率最高是 674 次注册，在 2014-05-13 这一天


# In[11]:


# 根据特征 date_account_created 可以分析用户随着时间的增长情况

# 首先计算数据集中到目前为止每一天有多少注册用户
dac_train = df_train.date_account_created.value_counts()

dac_test = df_test.date_account_created.value_counts()

dac_train[:5]


# In[12]:


# 将数据类型转换为 datatime 类型, 便于处理
dac_train_date = pd.to_datetime(df_train.date_account_created.value_counts().index)

dac_test_date = pd.to_datetime(df_test.date_account_created.value_counts().index)

dac_train_date


# In[13]:


# 计算用户注册时间 和 系统首次开放注册时间 之间相差的天数
dac_train_day = dac_train_date - dac_train_date.min()

dac_test_day = dac_test_date - dac_train_date.min()

dac_train_day


# In[14]:


# motplotlib 作图，进行可视化分析特征 date_account_created

# 创建绘制环境
fig = plt.figure()

# 创建一个图形环境
ax = fig.add_subplot(1, 1, 1)

# 指定位置进行绘制图形
ax.scatter(dac_train_day.days, dac_train.values, color = 'r', label = 'train dataset')
ax.scatter(dac_test_day.days, dac_test.values, color = 'b', label = 'test dataset')

# 添加图形的一些基本信息
ax.set_title("Accounts created vs day")
ax.set_xlabel("Days")
ax.set_ylabel("Accounts created")
ax.legend(loc = 'upper left')


# In[15]:


# 对上图的可视化可以知道：

# 1、x轴：代表着距离系统首次开放注册时间相差的天数
# 2、y轴：每天注册的用户数量
# 3、从图中可以知道，随着时间的增长, 用户注册的数量在急剧上升


# In[16]:


# 2 timestamp_first_active 用户首次活跃的时间，这个特征需要探索，转化一下

# 查看特征 timestamp_first_active 数据情况
df_train.timestamp_first_active.head()


# In[17]:


# 探索 时间戳 是否用重复的情况
df_train.timestamp_first_active.value_counts().unique()


# In[18]:


# 以上结果显示表明，时间戳没有重复的值，即就是 timestamp_first_active 特征没有重复

# 将 Unix 似的时间戳格式转换为 datetime 日期格式 年-月-日-时-分-秒
tfa_train_dt = df_train.timestamp_first_active.astype(str).apply(
               lambda x: datetime.datetime(int(x[:4]), int(x[4:6]), int(x[6:8]), 
                                           int(x[8:10]), int(x[10:12]), int(x[12:])))

tfa_train_dt


# In[19]:


# 查看特征数据 timestamp_first_active 的描述信息
tfa_train_dt.describe()


# In[20]:


# 根据描述信息可知

# 1、timestamp_first_active 特征没有重复值
# 2、timestamp_first_active 用户首次活跃时间 数据集开始于 2009-03-19
# 3、timestamp_first_active 用户首次活跃时间 数据集截止于 2014-06-30


# In[21]:


# 3 date_first_booking 用户首次预定的时间

# 查看 date_first_booking 数据情况
df_train.date_first_booking.describe()


# In[22]:


df_test.date_first_booking.describe()


# In[23]:


# 根据以上描述信息可知

# 1、train 数据集中 date_first_booking 特征大量缺失
# 2、test 数据集中 date_first_booking 特征完全缺失
# 3、date_first_booking 特征对于任务，几乎没有影响，考虑将这个特征直接删掉


# In[24]:


# 4 age 用户的年龄

# 查看 age 特征的数据情况
df_train.age.head()


# In[25]:


# 对 age 特征进行统计计数
df_train.age.value_counts()


# In[26]:


# 从以上显示结果可知
# 1、age 特征有缺失值
# 2、age 特征 的值由异常值
# 3、age 特征 用户年龄主要集中在 30 岁左右【28-32】


# In[27]:


# 对 age 特征 进行可视化分析

#首先将年龄进行分成 4 组: 缺失值，可能异常值（太大或者太小），合理值
# missing values, too small age, reasonable age, too large age
age_train =[df_train[df_train.age.isnull()].age.shape[0],
            df_train.query('age < 15').age.shape[0],
            df_train.query("age >= 15 & age <= 90").age.shape[0],
            df_train.query('age > 90').age.shape[0]]

age_test = [df_test[df_test.age.isnull()].age.shape[0],
            df_test.query('age < 15').age.shape[0],
            df_test.query("age >= 15 & age <= 90").age.shape[0],
            df_test.query('age > 90').age.shape[0]]

# age 年龄分组名
columns = ['Null', 'age < 15', 'age', 'age > 90']

# 可视化 age 特征，以便于分析了解数据情况
# 创建一个图形环境
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey = True,figsize=(10,5))

# 使用 seaborn 绘制柱状图
sns.barplot(columns, age_train, ax = ax1)
sns.barplot(columns, age_test, ax = ax2)

# 添加一些图形基本信息
ax1.set_title('training dataset')
ax2.set_title('test dataset')
ax1.set_ylabel('counts')


# In[28]:


# 对数据集中的其他特征字段，全部统一进行统计分析可视化

# 定义一个通用的函数，对数据集中数据进行可视化
def feature_barplot(feature, df_train=df_train, df_test=df_test, figsize=(10,5), rot = 90, saveimg = False): 
    # 对 特征字段 进行统计计数
    feat_train = df_train[feature].value_counts()
    feat_test = df_test[feature].value_counts()
    # 可视化 特征字段
    fig_feature, (ax1,ax2) = plt.subplots(1, 2, sharex=True, sharey = True, figsize = figsize)
    sns.barplot(feat_train.index.values, feat_train.values, ax = ax1)
    sns.barplot(feat_test.index.values, feat_test.values, ax = ax2)
    # 调整 刻度
    ax1.set_xticklabels(ax1.xaxis.get_majorticklabels(), rotation = rot)
    ax2.set_xticklabels(ax1.xaxis.get_majorticklabels(), rotation = rot)
    # 添加基本信息
    ax1.set_title(feature + ' of training dataset')
    ax2.set_title(feature + ' of test dataset')
    ax1.set_ylabel('Counts')
    # 调整图形布局
    plt.tight_layout()
    # 是否保存图片，默认不保存 False
    if saveimg == True:
        figname = feature + ".png"
        image_path = './../image/'
        fig_feature.savefig(image_path + figname, dpi = 75)


# In[29]:


# 5 gender 特征 用户性别
feature_barplot("gender", saveimg=True)


# In[30]:


# 通过可视化可知
# 1、用户性别，数据集中大量是未知的，少量是其他
# 2、male 男性 比 female 女性 要少一点，可以看做男女比例差不多
# 3、数据集中未知性别的数量和已知性别数量   从数量上基本是等价的
# 4、具体的比例信息，可以通过 计算小分组中的相对比例， 因此标准化分组百分比之和为 1，这样可能更加明确


# In[31]:


# 6 signup_method 特征 用户注册方式
feature_barplot("signup_method", saveimg=True)


# In[32]:


# 从可视化结果可知
# 1、用户注册方式，大部分是 basic 方式
# 2、少部分是 Facebook 方式
# 3、train 训练集中没有采用 google 方式，而 test 测试集中有采用 google 方式
# 4、具体的比例信息，可以通过 计算小分组中的相对比例， 因此标准化分组百分比之和为 1，这样可能更加明确


# In[33]:


#  7 signup_flow 特征 用户注册页面
feature_barplot("signup_flow", saveimg=True)


# In[34]:


# 8 language 特征 语言
feature_barplot("language", saveimg=True)


# In[35]:


# 9 affiliate_channel 特征 用户付费渠道
feature_barplot("affiliate_channel", saveimg=True)


# In[36]:


# 10  first_affiliate_tracked 特征 用户注册之前接触的第一付费渠道
feature_barplot("first_affiliate_tracked", saveimg=True)


# In[37]:


# 11 signup_app 特征 用户注册APP
feature_barplot("signup_app", saveimg=True)


# In[38]:


# 12 first_device_type 特征字段 用户移动设备类型
feature_barplot("first_device_type", saveimg=True)


# In[39]:


# 13 first_browser 特征字段 用户使用的浏览器类型
feature_barplot("first_browser", saveimg=True)


# In[40]:


# Loading the Data for session

# 加载 sessions 用户交互浏览记录数据集
df_sessions = pd.read_csv(dataset_path + 'sessions.csv.zip')

df_sessions.shape


# In[41]:


df_sessions[:5]


# In[42]:


# 为了将 sessions 数据集 和 train 数据集合并
# 需要经 user_id 特征字段 修改为 id

# 首先添加一列，即新增一个字段特征
df_sessions['id'] = df_sessions['user_id']

df_sessions[:5]


# In[43]:


# 删除 user_id 这一列
df_sessions = df_sessions.drop(['user_id'], axis=1) 

df_sessions[:5]


# In[44]:


# 查看数据集中的缺失值
df_sessions.isnull().sum()


# In[45]:


# 由以上的分析结果可知
# 1、 action 、 action_type 、action_detail 、secs_elapsed 这几个字段特征大量缺失
# 2、 device_type 这个字段特征没有缺失值
# 3、id 这个字段特征有一些缺失？？？


# In[46]:


# 填充缺失值
# 对 action 、 action_type 、action_detail 进行填充 NAN
# # secs_elapsed 用户停留时长，不能简单的填充 NAN，后续做填充处理
df_sessions.action = df_sessions.action.fillna('NAN')

df_sessions.action_type = df_sessions.action_type.fillna('NAN')

df_sessions.action_detail = df_sessions.action_detail.fillna('NAN')

df_sessions.isnull().sum()


# In[47]:


# 特征提取，对数据集有一定了解之后
# 需要针对 任务 对特征进行提取

# 对 sessions 数据集特征进行提取

# action 特征字段
df_sessions.action.head()


# In[48]:


df_sessions.action.value_counts().min()


# In[49]:


df_sessions.action.value_counts().max()


# In[51]:


# 以上结果显示，用户的交互行为至少发生一次
# 而且行为有多种形式

# 将特征 action 次数低于阈值 100 的列为 OTHER（other）
# Action values with low frequency are changed to 'OTHER'
# 设置阈值 100
act_freq = 100 

# 用户具体行为和频率一一对应为字典形式
# np.unique(df_sessions.action, return_counts=True) 取以数组形式返回非重复的 action值和它的数量
# zip（*（a,b））  a,b种元素一一对应，返回 zip object
act = dict(zip(*np.unique(df_sessions.action, return_counts=True)))

# 将低于 100 次的行为设定为 other
df_sessions.action = df_sessions.action.apply(lambda x: 'OTHER' if act[x] < act_freq else x)

df_sessions.action[:10]


# In[52]:


# 对特征 action，action_detail，action_type，device_type，secs_elapsed 进行细化

# 首先将用户的特征根据用户 id 进行分组
# 特征 action —— 统计每个用户总的 action 出现的次数，各个 action 类型的数量，平均值以及标准差
# 特征 action_detail —— 统计每个用户总的 action_detail 出现的次数，各个 action_detail 类型的数量，平均值以及标准差
# 特征 action_type —— 统计每个用户总的 action_type 出现的次数，各个 action_type 类型的数量，平均值，标准差以及总的停留时长（进行log处理）
# 特征 device_type —— 统计每个用户总的 device_type 出现的次数，各个 device_type 类型的数量，平均值以及标准差
# 特征 secs_elapsed —— 对缺失值用 0 填充，统计每个用户 secs_elapsed 时间的总和，平均值，
#                       标准差以及中位数（进行log处理），（总和/平均数），secs_elapsed（log处理后）各个时间出现的次数



# 对 action 特征进行细化，统计总的次数并排序
f_act = df_sessions.action.value_counts().argsort()

f_act_detail = df_sessions.action_detail.value_counts().argsort()

f_act_type = df_sessions.action_type.value_counts().argsort()

f_dev_type = df_sessions.device_type.value_counts().argsort()


# 按照用户 id 进行分组
dgr_sess = df_sessions.groupby(['id'])

# 对分组进行循环，得到所有特征
# Loop on dgr_sess to create all the features.
samples = []
ln = len(dgr_sess)
for g in dgr_sess:  # 对 dgr_sess 中每个 id 的数据进行遍历
    gr = g[1]
    
    l = []  # 建一个空列表，临时存放特征
    
    # the id    for example:'zzywmcn0jv'
    l.append(g[0]) #将 id 值放入空列表中
    
    # number of total actions
    l.append(len(gr)) # 将 id 对应数据的长度放入列表
    
    # secs_elapsed 特征中的缺失值用 0 填充再获取具体的停留时长值
    sev = gr.secs_elapsed.fillna(0).values  
    
    # action features 特征-用户行为 
    # 每个用户行为出现的次数，各个行为类型的数量，平均值以及标准差
    c_act = [0] * len(f_act)
    for i,v in enumerate(gr.action.values): # i是从 0-1对应的位置，v 是用户行为特征的值
        c_act[f_act[v]] += 1
    _, c_act_uqc = np.unique(gr.action.values, return_counts=True)
    # 计算用户行为行为特征各个类型数量的长度，平均值以及标准差
    c_act += [len(c_act_uqc), np.mean(c_act_uqc), np.std(c_act_uqc)]
    l = l + c_act
    
    # action_detail features 特征-用户行为具体
    # (how many times each value occurs, numb of unique values, mean and std)
    c_act_detail = [0] * len(f_act_detail)
    for i,v in enumerate(gr.action_detail.values):
        c_act_detail[f_act_detail[v]] += 1
    _, c_act_det_uqc = np.unique(gr.action_detail.values, return_counts=True)
    c_act_detail += [len(c_act_det_uqc), np.mean(c_act_det_uqc), np.std(c_act_det_uqc)]
    l = l + c_act_detail
    
    # action_type features  特征-用户行为类型 click等
    # (how many times each value occurs, numb of unique values, mean and std
    # + log of the sum of secs_elapsed for each value)
    l_act_type = [0] * len(f_act_type)
    c_act_type = [0] * len(f_act_type)
    for i,v in enumerate(gr.action_type.values):
        l_act_type[f_act_type[v]] += sev[i] # sev = gr.secs_elapsed.fillna(0).values ，求每个行为类型总的停留时长
        c_act_type[f_act_type[v]] += 1  
    l_act_type = np.log(1 + np.array(l_act_type)).tolist() # 每个行为类型总的停留时长，差异比较大，进行log处理
    _, c_act_type_uqc = np.unique(gr.action_type.values, return_counts=True)
    c_act_type += [len(c_act_type_uqc), np.mean(c_act_type_uqc), np.std(c_act_type_uqc)]
    l = l + c_act_type + l_act_type    
    
    # device_type features 特征-设备类型
    # (how many times each value occurs, numb of unique values, mean and std)
    c_dev_type  = [0] * len(f_dev_type)
    for i,v in enumerate(gr.device_type .values):
        c_dev_type[f_dev_type[v]] += 1 
    c_dev_type.append(len(np.unique(gr.device_type.values))) 
    _, c_dev_type_uqc = np.unique(gr.device_type.values, return_counts=True)
    c_dev_type += [len(c_dev_type_uqc), np.mean(c_dev_type_uqc), np.std(c_dev_type_uqc)]        
    l = l + c_dev_type    
    
    # secs_elapsed features  特征-停留时长     
    l_secs = [0] * 5 
    l_log = [0] * 15
    if len(sev) > 0:
        # Simple statistics about the secs_elapsed values.
        l_secs[0] = np.log(1 + np.sum(sev))
        l_secs[1] = np.log(1 + np.mean(sev)) 
        l_secs[2] = np.log(1 + np.std(sev))
        l_secs[3] = np.log(1 + np.median(sev))
        l_secs[4] = l_secs[0] / float(l[1]) #
        
        # Values are grouped in 15 intervals. Compute the number of values
        # in each interval.
        # sev = gr.secs_elapsed.fillna(0).values 
        log_sev = np.log(1 + sev).astype(int)
        # np.bincount():Count number of occurrences of each value in array of non-negative ints.  
        l_log = np.bincount(log_sev, minlength=15).tolist()                    
    l = l + l_secs + l_log
    
    # The list l has the feature values of one sample.
    samples.append(l)

# preparing objects    
samples = np.array(samples) 
samp_ar = samples[:, 1:].astype(np.float16) # 取除 id 外的特征数据
samp_id = samples[:, 0]   # 取id，id位于第一列

# 为提取的特征创建一个dataframe     
col_names = []    # name of the columns
for i in range(len(samples[0])-1):  # 减1的原因是因为有个id
    col_names.append('c_' + str(i))  # 起名字的方式    
df_agg_sess = pd.DataFrame(samp_ar, columns=col_names)
df_agg_sess['id'] = samp_id
df_agg_sess.index = df_agg_sess.id # 将id作为 index


df_agg_sess.head()


# In[54]:


# 通过对 sessions 数据集的特征提取
# session 数据集由 6 个特征变为 458 个特征
df_agg_sess.describe()


# In[70]:


# 对 trian 和 test 数据集进行特征提取
# 标记 train 数据集的行数和存储进行预测的目标变量
# labels 存储了进行预测的目标变量 country_destination

train = pd.read_csv(dataset_path + "train_users_2.csv.zip")
test = pd.read_csv(dataset_path + "test_users.csv.zip")

# 计算出 train 的行数，便于之后对 train 和 test 数据进行分离操作
train_row = train.shape[0]

# The label we need to predict
labels = train['country_destination'].values


# 删除 date_first_booking 和 train 文件中的 country_destination
# 数据探索时发现 date_first_booking 在 train 和 test 文件中缺失值太多，故删除
# 删除 country_destination，用模型预测 country_destination，再与已经存储 country_destination 的 labels 进行比较，从而判断模型优劣
train.drop(['country_destination', 'date_first_booking'], axis = 1, inplace = True)
test.drop(['date_first_booking'], axis = 1, inplace = True)


# 合并 train 和 test 数据集
# 便于进行相同的特征提取操作
df = pd.concat([train, test], axis = 0, ignore_index = True)

# 其他特征处理
# 在数据探索时，发现剩余的特征 lables 都比较少，故不进一步进行特征提取，只进行 one-hot-encoding 处理
feat_toOHE = ['gender', 
             'signup_method', 
             'signup_flow', 
             'language', 
             'affiliate_channel', 
             'affiliate_provider', 
             'first_affiliate_tracked', 
             'signup_app', 
             'first_device_type', 
             'first_browser']

# 对其他特征进行 one-hot-encoding 处理
for f in feat_toOHE:
    df_ohe = pd.get_dummies(df[f], prefix=f, dummy_na=True)
    df.drop([f], axis = 1, inplace = True)
    df = pd.concat((df, df_ohe), axis = 1)


# In[71]:


# timestamp_first_active 将时间戳格式转换为 datetime 类型
tfa = df.timestamp_first_active.astype(str).apply(
                                lambda x: datetime.datetime(int(x[:4]), int(x[4:6]), int(x[6:8]),
                                                            int(x[8:10]), int(x[10:12]), int(x[12:])))

# 提取 年 月 日
# create tfa_year, tfa_month, tfa_day feature
df['tfa_year'] = np.array([x.year for x in tfa])
df['tfa_month'] = np.array([x.month for x in tfa])
df['tfa_day'] = np.array([x.day for x in tfa])


# 提取特征：weekday 并对结果进行 one hot encoding 编码
# isoweekday() 可以返回一周的星期几，e.g.星期日：0；星期一：1
df['tfa_wd'] = np.array([x.isoweekday() for x in tfa])

# one hot encoding
df_tfa_wd = pd.get_dummies(df.tfa_wd, prefix = 'tfa_wd')

# 添加 df['tfa_wd'] 编码后的特征
df = pd.concat((df, df_tfa_wd), axis = 1)

# 删除原有未编码的特征
df.drop(['tfa_wd'], axis = 1, inplace = True)



# 提取特征：季节   因为判断季节关注的是月份，故对年份进行统一
Y = 2000
seasons = [(0, (date(Y,  1,  1),  date(Y,  3, 20))),  # 'winter'
           (1, (date(Y,  3, 21),  date(Y,  6, 20))),  # 'spring'
           (2, (date(Y,  6, 21),  date(Y,  9, 22))),  # 'summer'
           (3, (date(Y,  9, 23),  date(Y, 12, 20))),  # 'autumn'
           (0, (date(Y, 12, 21),  date(Y, 12, 31)))]  # 'winter'

def get_season(dt):
    dt = dt.date() # 获取日期
    dt = dt.replace(year=Y) # 将年统一换成 2000 年
    return next(season for season, (start, end) in seasons if start <= dt <= end)

df['tfa_season'] = np.array([get_season(x) for x in tfa])

# one hot encoding 
df_tfa_season = pd.get_dummies(df.tfa_season, prefix = 'tfa_season')
df = pd.concat((df, df_tfa_season), axis = 1)
df.drop(['tfa_season'], axis = 1, inplace = True)


# In[72]:


# date_account_created    
# 将 date_account_created 转换为 datetime 类型
dac = pd.to_datetime(df.date_account_created)

# 提取 年 月 日
# create year, month, day feature for dac
df['dac_year'] = np.array([x.year for x in dac])
df['dac_month'] = np.array([x.month for x in dac])
df['dac_day'] = np.array([x.day for x in dac])

# 提取 weekday
# create features of weekday for dac
df['dac_wd'] = np.array([x.isoweekday() for x in dac])
df_dac_wd = pd.get_dummies(df.dac_wd, prefix = 'dac_wd')
df = pd.concat((df, df_dac_wd), axis = 1)
df.drop(['dac_wd'], axis = 1, inplace = True)


# 提取季节
# create season features fro dac
df['dac_season'] = np.array([get_season(x) for x in dac])
df_dac_season = pd.get_dummies(df.dac_season, prefix = 'dac_season')
df = pd.concat((df, df_dac_season), axis = 1)
df.drop(['dac_season'], axis = 1, inplace = True)


# 提取特征：date_account_created 和 timestamp_first_active 之间的差值
# 即用户在 airbnb 平台活跃到正式注册所花的时间
dt_span = dac.subtract(tfa).dt.days 

# 查看 dt_span 的头十行数据
dt_span.value_counts().head(10)


# In[73]:


# 以上结果显示可知分析
#      数据主要集中在 -1，可以猜测，用户当天注册dt_span值便是-1

# 从差值提取特征：差值为一天，一月，一年和其他
# 即用户活跃到注册花费的时间为一天，一月，一年或其他
# create categorical feature: span = -1; -1 < span < 30; 31 < span < 365; span > 365
def get_span(dt):
    # dt is an integer
    if dt == -1:
        return 'OneDay'
    elif (dt < 30) & (dt > -1):
        return 'OneMonth'
    elif (dt >= 30) & (dt <= 365):
        return 'OneYear'
    else:
        return 'other'

df['dt_span'] = np.array([get_span(x) for x in dt_span])
df_dt_span = pd.get_dummies(df.dt_span, prefix = 'dt_span')
df = pd.concat((df, df_dt_span), axis = 1)
df.drop(['dt_span'], axis = 1, inplace = True)


# 删除原有的特征
# 对 timestamp_first_active，date_account_created 进行特征提取后，从特征列表中删除原有的特征
df.drop(['date_account_created','timestamp_first_active'], axis = 1, inplace = True)


# In[74]:


# age 特征字段
# Age 获取年龄
av = df.age.values

# 在数据探索阶段，发现大部分数据是集中在（15，90）区间的，
# 但有部分年龄分布在（1900，2000）区间，
# 猜测用户是把出生日期误填为年龄，故进行预处理
# This are birthdays instead of age (estimating age by doing 2014 - value)
# 数据来自2014年，故用 2014 - value
av = np.where(np.logical_and(av<2000, av>1900), 2014-av, av)

df['age'] = av


# 将年龄进行分段
# Age has many abnormal values that we need to deal with. 
age = df.age
age.fillna(-1, inplace = True) #空值填充为-1
div = 15
def get_age(age):
    # age is a float number  将连续型转换为离散型
    if age < 0:
        return 'NA' #表示是空值
    elif (age < div):
        return div #如果年龄小于15岁，那么返回15岁
    elif (age <= div * 2):
        return div*2 #如果年龄大于15小于等于30岁，则返回30岁
    elif (age <= div * 3):
        return div * 3
    elif (age <= div * 4):
        return div * 4
    elif (age <= div * 5):
        return div * 5
    elif (age <= 110):
        return div * 6
    else:
        return 'Unphysical' #非正常年龄


# 将分段后的年龄作为新的特征放入特征列表中
df['age'] = np.array([get_age(x) for x in age])
df_age = pd.get_dummies(df.age, prefix = 'age')
df = pd.concat((df, df_age), axis = 1)
df.drop(['age'], axis = 1, inplace = True)


# In[95]:


# 将对 session 提取的特征整合到一起
df_all = pd.merge(df, df_agg_sess, how='left', on='id')
df_all = df_all.drop(['id'], axis=1) # 删除id
df_all = df_all.fillna(-2)  # 对没有 sesssion data 的特征进行缺失值处理

#加了一列，表示每一行总共有多少空值，这也作为一个特征
df_all['all_null'] = np.array([sum(r<0) for r in df_all.values]) 


# In[96]:


# 数据准备

# 将 train 和 test 数据进行分离操作
# train_row 是之前记录的 train 数据行数
Xtrain = df_all.iloc[:train_row, :]
Xtest = df_all.iloc[train_row:, :]


# 将提取的特征生成 CSV 文件保存
Xtrain.to_csv(dataset_path + "Airbnb_xtrain_v2.csv")
Xtest.to_csv(dataset_path + "Airbnb_xtest_v2.csv")

# labels.tofile（）：Write array to a file as text or binary (default)
labels.tofile(dataset_path + "Airbnb_ytrain_v2.csv", sep='\n', format='%s') # 存放目标变量

# 读取特征文件
xtrain = pd.read_csv(dataset_path + "Airbnb_xtrain_v2.csv", index_col=0)
ytrain = pd.read_csv(dataset_path + "Airbnb_ytrain_v2.csv", header=None)

xtrain.head()


# In[97]:


ytrain.head()


# In[98]:


# 分析：可以发现经过特征提取后特征文件 xtrain 扩展为 665 个特征，ytrain 中包含训练集中的目标变量
# # 将目标变量进行 labels encoding

# labels encoding前：
# [‘AU’, ‘CA’, ‘DE’, ‘ES’, ‘FR’, ‘GB’, ‘IT’, ‘NDF’, ‘NL’, ‘PT’, ‘US’,‘other’]
# labels encoding后：
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

le = LabelEncoder()
ytrain_le = le.fit_transform(ytrain.values)


# 提取 10% 进行模型训练
# Let us take 10% of the data for faster training. 
n = int(xtrain.shape[0]*0.1)
xtrain_new = xtrain.iloc[:n, :]  # 训练数据
ytrain_new = ytrain_le[:n]       # 训练数据的目标变量



# StandardScaling the dataset
# 标准缩放数据集
# Standardization of a dataset is a common requirement for many machine learning estimators: 
# they might behave badly if the individual feature do not more or less look like standard normally distributed data
# (e.g. Gaussian with 0 mean and unit variance)
# 数据集标准化是许多机器学习估计器的普遍要求：
# 如果单个特征或多或少看起来不像标准正态分布数据（例如均值和单位方差为0的高斯），它们可能表现不佳。
X_scaler = StandardScaler()
xtrain_new = X_scaler.fit_transform(xtrain_new)


# In[99]:


# 评测指标 DNCG

# 1、NDCG 是一种衡量排序质量的评价指标，该指标考虑了所有元素的相关性
# 2、由于预测的目标变量并不是二分类变量，故用 NDGG 模型来进行模型评分，判断模型优劣

from sklearn.metrics import make_scorer

def dcg_score(y_true, y_score, k=5):
    
    """
    y_true : array, shape = [n_samples] # 数据
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes] # 预测的分数
        Predicted scores.
    k : int
    """
    order = np.argsort(y_score)[::-1] # 分数从高到低排序
    y_true = np.take(y_true, order[:k]) # 取出前 k[0,k）个分数
      
    gain = 2 ** y_true - 1   

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)
  

def ndcg_score(ground_truth, predictions, k=5):   

    """
    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes] 
        Predicted probabilities. 预测的概率
    k : int
        Rank.
    """
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)    
    scores = []
    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)


# In[ ]:




