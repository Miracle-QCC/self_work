import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsRegressor


rc={'font.sans-serif':'SimHei','axes.unicode_minus':False}
seaborn.set(context='notebook',style='ticks',rc=rc)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 对输入向量进行减去最大值的操作，防止指数溢出
    return exp_x / np.sum(exp_x, axis=0)


#
def get_evaluation(y_pred, y_true):
    """
    获得评价指标
    :param y_pred:
    :param y_true:
    :param columns:
    :return:
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"出站人数 MSE:{mse},RMSE:{rmse},MAE:{mae},MAPE:{mape}")


def plot_result(y_pred,y_true,d):
    """
    画真实值和预测值的曲线
    :param y_pred:
    :param y_true:
    :param columns:
    :return:
    """
    # plt.figure(figsize=(8,8))
    # for i in range(len(y_pred)):
        # plt.subplot(2,2,i+1)
    plt.plot(range(1,len(y_pred)+1),y_pred,"r",label="pred")
    plt.plot(range(1,len(y_true)+1),y_true,"g",label="true")
    plt.title(f"{d} pred-true")
    plt.ylabel(f"进展人数")
    # plt.xticks(tt[0:len(y_pred):20], rotation="90", fontsize=8)
    plt.legend()
    plt.savefig(f"pred_true{k}.jpg")
    plt.clf()  # 或者 plt.cla()

    # plt.show

def get_pred(x, y):
    """

    :param x: 1*k ,数据与其他的数据的距离，其中最近的k个
    :param y: 1*k, 最近的k个的y
    :return: 1*1 ，返回预测值
    """
    weight = softmax(x)
    y_pred = (y.squeeze() * weight).sum()
    return y_pred

def loss_fun(y,y_):
    return mean_squared_error(y,y_)
    # mae = abs(y - y_).mean()
    # return mae

class SEKNN:

    def __init__(self, ratio=0.5):
        self.ratio=ratio  # 进出站流量距离的比例:ratio,变化率距离比例:(1-ratio)
        self.datasets=None  # 数据集

    @staticmethod
    def distance(x1, x2):  # 欧几里得距离
        return np.square(x1 - x2)

    def fit(self,x, norm_x, y):  # 训练，KNN为懒惰学习，直接把训练集当作成员属性
        """
        :param datasets: nx5 ['time', '出站(5月1日)', '进站(5月1日)', '变化率（出）', '变化率（进）']
        :return:
        """
        self.datasets_x = x
        self.datasets_y = y
        self.norm_x = norm_x

    def predict(self, norm_x, y, k):  # 对数据进行预测

        """
       :param norm_x: 经过归一化的x，计算距离 nx4 ['进展人数', '变化率（出）', '变化率（进）']
       :param norm_y: 经过归一化的y，计算距离
       :param k: KNNd的K
       :return:
       """

        ### 计算两两向量之间的欧氏距离
        euclidean_distances = cdist(norm_x, self.norm_x)
        ### 计算两两向量之间的欧氏距离
        cosine_distances = 1 - np.dot(norm_x, self.norm_x.T) / (
                np.linalg.norm(norm_x, axis=1)[:, np.newaxis] * np.linalg.norm(self.norm_x, axis=1))


        ### 整合两种距离
        distances = self.ratio * euclidean_distances + (1 - self.ratio) * cosine_distances

        ### 选择最近的k个
        closest_indices = np.argsort(distances, axis=1)[:,:k]

        # loss = 0
        y_preds = []
        for i in range(len(distances)):
            y_pred = get_pred(distances[i,closest_indices[i]], self.datasets_y[closest_indices[i]])
            y_preds.append(y_pred)
            # y_gt = y[i]
            # loss += loss_fun(y_pred, y_gt)
            # print(loss_fun(y_pred, y_gt))
        # print(loss)
        plot_result(y_preds, y, date)
        y_preds = np.array(y_preds)
        loss = mean_absolute_error(y_preds, y)
        # print(loss)
        get_evaluation(y_preds,y)
        return loss


def get_datasets(d,root="1_content_1693294875568.xlsx"):

    # 获取数据集
    """

    :param d: 测试集，在建立训练集时剔除掉
    :param root: 源文件
    :return: 返回训练集（输入、标签）、测试集（输入、标签）
    """
    columns = pd.read_excel(root, sheet_name="5.1").columns
    data_train = []
    for i in [item for item in  [1, 2, 3, 4, 5, 6, 7, 8,9,10,12,13] if item!=int(d.split(".")[-1])]:
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
    train_data = data_train.values
    # 划分输入和输出
    x_train = np.concatenate((train_data[:,1:][:,:1], train_data[:,1:][:,2:]),axis=1)
    y_train = train_data[:,2].reshape(-1,1)

    ### 提取测试集，选定d作为测试集
    data_test = []
    data_temp = pd.read_excel("1_content_1693294875568.xlsx", sheet_name=d)
    data_temp.rename(columns={"Unnamed: 0": "time"}, inplace=True)
    data_temp["time"] = data_temp["time"].astype(str)
    date = f"2023-05-{i if i > 9 else f'0{i}'}"
    data_temp["time"] = data_temp["time"].apply(lambda x: date + " " + x)
    data_test.append(data_temp.values)
    data_test = np.concatenate(data_test, axis=0)
    data_test = pd.DataFrame(data_test, columns=columns)
    data_test.rename(columns={"Unnamed: 0": "time"}, inplace=True)  # 转换列名
    data_test["time"] = data_test["time"].astype(str)

    test_data = data_test.values
    # 划分输入和输出
    x_test = np.concatenate((test_data[:, 1:][:, :1], test_data[:, 1:][:, 2:]), axis=1)
    y_test = test_data[:, 2].reshape(-1, 1)


    return (x_train, y_train), (x_test, y_test)

def normlize_data(data, ma, mi):
    """
    使用最大最小归一化
    :param data: 需要归一化的数据
    :param ma: 收集到的最大值
    :param mi: 收集到的最小值
    :return:
    """
    norm_data = (data - mi) / (ma - mi)
    return norm_data



def loss(y_pred, y):
    return np.sqrt(((y_pred - y) ** 2).mean())

if __name__ == '__main__':
    date="5.9"
    """
    本次实验利用改进的KNN算法进行回归预测，
    输入信息为出站人数、变化率（出）、变化率（进），输出为进站人数
    # """
    losses = []
    for k in tqdm(range(1,50)):
        ratio=0.5
        # print("k:", k)
        datasets_train, datasets_test = get_datasets(date)
        x_train, y_train = datasets_train
        x_test, y_test = datasets_test
        x_train = x_train.astype(float)
        y_train = y_train.astype(float)
        x_test = x_test.astype(float)
        y_test = y_test.astype(float)

        ##  记录最大最小值，用于归一化
        # x_ma = np.max(x_train, axis=0)
        # x_mi = np.min(x_train, axis=0)
        # norm_x_train = normlize_data(x_train,x_ma,x_mi).astype(float)
        # norm_x_test = normlize_data(x_test, x_ma, x_mi).astype(float)

        # datasets_date=get_test_datasets(date=date)


        model_knn = SEKNN(ratio=ratio)

        ###  将历史数据存入
        model_knn.fit(x_train, x_train, y_train)
        #  预测，计算loss
        loss = model_knn.predict(x_test, y_test, k)

        losses.append(loss)

    X = range(1,50)
    print("最小loss：",min(losses))
    # 绘制折线图
    plt.plot(X, losses, label='loss')

    # 添加标题和标签
    plt.title('Loss variation chart')
    plt.xlabel('k')
    plt.ylabel('loss')

    # 显示图例
    plt.legend()

    # 显示图形
    # plt.show()
    plt.savefig("SEKNN的loss随K变化曲线")


