# @Author : huzejun
# @Time : 2023/3/2 3:21

from sklearn.datasets import load_iris

def datasets_demo():

    iris = load_iris()
    print("鸢尾花:\n",iris)
    print("查看数据集描述:\n",iris["DESCR"])
    print("查看特征值的名字:\n",iris.feature_names)
    print("查看特征值:\n",iris.data,iris.data.shape)

    return None

if __name__ == "__main__":

    datasets_demo()



