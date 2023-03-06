# @Author : huzejun
# @Time : 2023/3/2 3:21

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import jieba
import pandas as pd

def datasets_demo():

    iris = load_iris()
    print("鸢尾花:\n",iris)
    print("查看数据集描述:\n",iris["DESCR"])
    print("查看特征值的名字:\n",iris.feature_names)
    print("查看特征值:\n",iris.data,iris.data.shape)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值:\n",x_train,x_train.shape)
    return None


def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'city': '北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}]
    # 1、实例化一个转换器类
    transfer = DictVectorizer(sparse=True)

    # 2、调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray(), type(data_new))
    print("特征名字:\n", transfer.get_feature_names_out())

    return None

def count_demo():
    """
    文本特征抽取：CountVecotrizer
    :return:
    """
    data = ["lief is short,i like like python","life is too long, i dislike python"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["is","too"])

    # 2、调用 fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

def count_chinese_demo():
    """
    中文文本特征抽取：CountVecotrizer
    :return:
    """
    data = ["我 爱 北京 天安门","天安门 上 太阳 升"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer()

    # 2、调用 fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

def cut_word(text):
    """
    进行中文分词："我爱北京天安门" "我 爱 北京 天安门"
    :param text:
    :return:
    """
    # text = " ".join(list(jieba.cut(text)))
    # # print(type(a))
    # print(text)
    # return text

    return " ".join(list(jieba.cut(text)))

def count_chinese_demo2():
    """
    中文文本特征抽取，自动分词
    :return:
    """
    # 1、将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事务，你就不会真正了解它。了解事务真正含义的秘密取决于如何将其与我们所了解的事务相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["一种","所以","不要"])

    # 2、调用 fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n",data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

def tfidf_demo():
    """
    用 TF-idf文本特征抽取
    :return:
    """
    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事务，你就不会真正了解它。了解事务真正含义的秘密取决于如何将其与我们所了解的事务相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1、实例化一个转换器类
    transfer = TfidfVectorizer(stop_words=["一种","所以"])

    # 2、调用 fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n",data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

def minmax_demo():
    """
    归一化
    :return:
    """
    # 1、获取数据
    data = pd.read_csv("dating.txt", sep='\t') # datingTestSet.txt
    # data = pd.read_csv("datingTestSet.txt") # datingTestSet.txt
    # data = pd.read_csv('datingTestSet2.txt', sep='\t') # datingTestSet2.txt

    data = data.iloc[:, :3]
    print("data:\n",data)

    # 2、实例化一个转换器
    transfer = MinMaxScaler()
    # transfer = MinMaxScaler(feature_range=[1, 2])

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return None

def stand_demo():
    """
    标准化

    :return:
    """
    # 1、获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    print("data:\n",data)

    # 2、实例化一个转换器
    transfer = StandardScaler()

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return None

def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1、获取数据
    data = pd.read_csv("factor_returns.csv")
    data = data.iloc[:, 1:-2]
    print("data\n",data)

    # 2、实例化一个转换器类
    transfer = VarianceThreshold(threshold=10)

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new,data_new.shape)

    # 计算某两个变量之间的相关系数
    r1 = pearsonr(data["pe_ratio"],data["pb_ratio"])
    print("相关系数：\n", r1)
    r2 = pearsonr(data["revenue"],data["total_expense"])
    print("revenue与total_expense相关性：\n", r2)

    return None



if __name__ == "__main__":
    # 代码1：sklearn数据集使用
    # datasets_demo()

    # 代码2：字典特征抽取
    # dict_demo()

    # 文本特征抽取：CountVecotrizer
    # count_demo()
    # 中文文本特征抽取：CountVecotrizer
    # count_chinese_demo()
    # 中文文本特征抽取，自动分词
    # count_chinese_demo2()
    # 代码6 中文分词

    # print(cut_word("我爱北京天安门"))
    # 代码7     用 TF-idf文本特征抽取
    # tfidf_demo()
    # 代码8   归一化
    # minmax_demo()
    # 代码9    标准化
    # stand_demo()
    # 代码10  过滤低方差特征
    variance_demo()
