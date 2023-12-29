import os
import pickle
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
rc={'font.sans-serif':'SimHei','axes.unicode_minus':False}
seaborn.set(context='notebook',style='ticks',rc=rc)

def mean_absolute_percentage_error_(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6)) * 100)

class KNN(object):

    def __init__(self,ratio=0.5):
        self.ratio=ratio  # 进出站流量距离的比例:ratio,变化率距离比例:(1-ratio)
        self.datasets=None  # 数据集

    @staticmethod
    def l2_distance(x1,x2):  # 欧几里得距离
        return np.sqrt(np.sum(np.square(x1-x2)))

    @staticmethod
    def distance(x1, x2):  # 欧几里得距离
        return np.square(x1 - x2)

    def fit(self,datasets):  # 训练，KNN为懒惰学习，直接把训练集当作成员属性
        """
        :param datasets: nx5 ['time', '出站(5月1日)', '进站(5月1日)', '变化率（出）', '变化率（进）']
        :return:
        """
        self.datasets=datasets

    def predict(self,x,k):  # 对数据进行预测
        """
        :param x: nx4 ['出站(5月1日)', '进站(5月1日)', '变化率（出）', '变化率（进）']
        :return: n*4
        """
        if self.datasets is None:
            raise ValueError("model doesn't fit!")
        print("training....")
        results=np.zeros((len(x),4))  # results用来保存结果
        for i in tqdm(range(len(x))):  # 遍历每条数据
            time=x[i,0]  # 时间
            flow_rate=x[i,1:3]  # 进出流量
            flow_percent=x[i,3:]  # 进出变化率
            distances_rate1=np.zeros(len(self.datasets))   # 保存距离
            distances_rate2 = np.zeros(len(self.datasets))  # 保存距离
            distances_percent1 = np.zeros(len(self.datasets)) # 保存距离
            distances_percent2 = np.zeros(len(self.datasets)) # 保存距离
            time_weights=np.zeros(len(self.datasets))   # 保存时间距离
            for j in range(len(self.datasets)):  # 遍历datasets,得到距离
                ### 如果用自身参与预测，那么其他点都是干扰，K越大则效果越差
                if i == j:
                    continue
                time_dis=(pd.to_datetime(time)-pd.to_datetime(self.datasets[j,0])).seconds/60/15  # 计算时间距离转换成分钟，然后/15
                flow_rate_dis1=self.distance(flow_rate[0],self.datasets[j,1])  # 得到流量距离
                flow_rate_dis2=self.distance(flow_rate[1],self.datasets[j,2])  # 得到流量距离

                flow_percent_dis1=self.distance(flow_percent[0],self.datasets[j,3])  # 得到变化率距离
                flow_percent_dis2=self.distance(flow_percent[1],self.datasets[j,4])  # 得到变化率距离

                distances_rate1[j]=flow_rate_dis1  # 保存距离
                distances_rate2[j]=flow_rate_dis2  # 保存距离

                distances_percent1[j]=flow_percent_dis1 # 保存距离
                distances_percent2[j]=flow_percent_dis2 # 保存距离

                ### 时间没有参与计算
                time_weights[j]=time_dis  # 保存时间距离

            ### 避免自身参与计算，并且占比过高
            distances_rate1[i] = np.max(distances_rate1) + 1
            distances_percent1[i] = np.max(distances_percent1) + 1
            distances_rate2[i] = np.max(distances_rate2) + 1
            distances_percent2[i] = np.max(distances_percent2) + 1

            ### 原来这里范围归一化为0 ~ 2
            # distances_rate = (distances_rate - np.min(distances_rate)) /(np.max(distances_rate) - np.min(distances_rate))  # 预先做一次归一化，不然exp之后的数据太小了
            distances_rate1 = (distances_rate1 - np.min(distances_rate1)) /(np.max(distances_rate1) - np.min(distances_rate1))  # 预先做一次归一化，不然exp之后的数据太小了
            distances_rate2 = (distances_rate2 - np.min(distances_rate2)) /(np.max(distances_rate2) - np.min(distances_rate2))  # 预先做一次归一化，不然exp之后的数据太小了

            ## 这一行代码有误
            # distances_percent = (distances_percent - np.min(distances_percent)) /(np.max(distances_rate) - np.min(distances_rate))
            distances_percent1 = (distances_percent1 - np.min(distances_percent1)) /(np.max(distances_percent1) - np.min(distances_percent1))
            distances_percent2 = (distances_percent2 - np.min(distances_percent2)) /(np.max(distances_percent2) - np.min(distances_percent2))



            # distances=self.ratio*distances_rate+(1-self.ratio)*distances_percent  # 综合距离的权重，ratio*流量距离+(1-ratio)*变化率距离
            distances = np.sqrt(distances_rate1 + distances_rate2 + distances_percent1 + distances_percent2) # 综合距离的权重，ratio*流量距离+(1-ratio)*变化率距离
            #distances *= 2

            dis_sort_index=np.argsort(distances)[:k]  # 取距离最小的top k个，计算索引
            distances=distances[dis_sort_index]  # 得到最小的top k个距离
            distances=np.exp(-2.0*np.square(distances))
            #distances= np.exp(-distances ** 2 / (2 * 1.33 ** 2))


            distances/=np.sum(distances)

            #time_weights/=np.sum(time_weights)  # 时间距离归一化
            # 得到一条数据的加权平均结果
            results[i,0]=np.sum(distances*self.datasets[dis_sort_index,1])
            results[i,1]=np.sum(distances*self.datasets[dis_sort_index,2])
            results[i,2]=np.sum(distances*self.datasets[dis_sort_index,3])
            results[i,3]=np.sum(distances*self.datasets[dis_sort_index,4])

        return results


def get_datasets(d,root="1_content_1693294875568.xlsx"):
    # 获取数据集

    columns = pd.read_excel(root, sheet_name="5.1").columns
    data_train = []
    for i in [item for item in [1, 2, 3, 4, 5, 6, 7, 8,9,10,12,13] if item!=int(d.split(".")[-1])]:
        data_temp = pd.read_excel("1_content_1693294875568.xlsx", sheet_name=f"5.{i}")
        data_temp.rename(columns={"Unnamed: 0": "time"}, inplace=True)
        data_temp["time"] = data_temp["time"].astype(str)
        date = f"2023-05-{i if i > 9 else f'0{i}'}"
        data_temp["time"] = data_temp["time"].apply(lambda x: date + " " + x)
        data_train.append(data_temp.values)
    data_train = np.concatenate(data_train, axis=0)
    data_train = pd.DataFrame(data_train, columns=columns)
    data_train.rename(columns={"Unnamed: 0":"time"},inplace=True)  # 转换列名
    data_train["time"]=data_train["time"].astype(str)
    x_train=data_train.values

    data_test = []
    for i in [item for item in [9,10,12,13] if item != int(d.split(".")[-1])]:
        data_temp = pd.read_excel("1_content_1693294875568.xlsx", sheet_name=f"5.{i}")
        data_temp.rename(columns={"Unnamed: 0": "time"}, inplace=True)
        data_temp["time"] = data_temp["time"].astype(str)
        date = f"2023-05-{i if i > 9 else f'0{i}'}"
        data_temp["time"] = data_temp["time"].apply(lambda x: date + " " + x)
        data_test.append(data_temp.values)
    data_test = np.concatenate(data_test, axis=0)
    data_test = pd.DataFrame(data_test, columns=columns)
    data_test.rename(columns={"Unnamed: 0": "time"}, inplace=True)  # 转换列名
    data_test["time"] = data_test["time"].astype(str)
    x_test = data_test.values

    return x_train,x_test


def get_test_datasets(date="5.1"):
    i=int(date.split(".")[-1])
    data=pd.read_excel("1_content_1693294875568.xlsx",sheet_name=date)
    data.rename(columns={"Unnamed: 0": "time"}, inplace=True)
    date = f"2023-05-{i if i > 10 else f'0{i}'}"
    data["time"] = data["time"].astype(str)
    data["time"] = data["time"].apply(lambda x: date + " " + x)
    return data.values


def plot_result(y_pred,y_true,columns,d):
    """
    画真实值和预测值的曲线
    :param y_pred:
    :param y_true:
    :param columns:
    :return:
    """
    plt.figure(figsize=(8,8))
    for i in range(len(columns)):
        plt.subplot(2,2,i+1)
        plt.plot(range(1,len(y_pred)+1),y_pred[:,i],"r",label="pred")
        plt.plot(range(1,len(y_true)+1),y_true[:,i],"g",label="true")
        plt.title(f"{d} {columns[i]} pred-true")
        plt.ylabel(f"{columns[i]}")
        # plt.xticks(tt[0:len(y_pred):20], rotation="90", fontsize=8)
        plt.legend()
    plt.savefig(f"pred_true{k}.jpg")
    # plt.show


def get_evaluation(y_pred,y_true,columns):
    """
    获得评价指标
    :param y_pred:
    :param y_true:
    :param columns:
    :return:
    """
    for i in range(len(columns)):
        mse=mean_squared_error(y_true[:,i],y_pred[:,i])
        rmse = np.sqrt(mse)
        mae=mean_absolute_error(y_true[:,i],y_pred[:,i])
        mape=mean_absolute_percentage_error_(y_true[:,i],y_pred[:,i])
        print(f"{columns[i]} MSE:{mse},RMSE:{rmse},MAE:{mae},MAPE:{mape}")


if __name__ == '__main__':
    date="5.9"
    # k=30
    for k in range(1,50):
        ratio=0.5
        print("k:", k)
        datasets_train,datasets_test=get_datasets(date)
        datasets_date=get_test_datasets(date=date)
        model_knn=KNN(ratio=ratio)
        model_knn.fit(datasets_train)
        res_train=model_knn.predict(datasets_train,k)

        model_knn.fit(datasets_test)
        res_test=model_knn.predict(datasets_test,k)

        plot_result(res_test,datasets_test[:,1:],columns=['出站(5月1日)', '进站(5月1日)', '变化率（出）', '变化率（进）'],d=date)
        print("测试集:")
        get_evaluation(res_test, datasets_test[:, 1:], columns=['出站(5月1日)', '进站(5月1日)', '变化率（出）', '变化率（进）'])
        print("训练集:")
        get_evaluation(res_train, datasets_train[:, 1:], columns=['出站(5月1日)', '进站(5月1日)', '变化率（出）', '变化率（进）'])




