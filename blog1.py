from sklearn.datasets import load_boston,load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold

#导入数据集
boston = load_boston()
iris = load_iris()

###预处理###

# 对列标准版
s_data=StandardScaler().fit_transform(boston.data)
print(s_data[0:5,:])


# 对列进行缩放,(0~1)
m_data=MinMaxScaler().fit_transform(boston.data)
print(m_data[0:5,:])


# 对行归一化，使得每个样本都变成单位向量
n_data=Normalizer().fit_transform(boston.data)
print(n_data[0:5,:])


# 二值化，阈值设定为 0.5 ，返回二值化的数据
b_data=Binarizer(threshold=0.5).fit_transform(boston.data)
print(b_data[0:5,:])


# 哑编码，对boston数据集的目标值，返回值为哑编码后的数据
o_target=OneHotEncoder().fit_transform(boston.target)
print(o_target[0:5])



###特征选择###


#方差选择法，返回值为特征选择后的数据
#参数threshold为方差的阈值
VarianceThreshold(threshold=3).fit_transform(iris.data)

# 卡方检验，选择K个最好的特征，返回选择特征后的数据
select_data=SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)

# 递归特征消除法，返回特征选择后的数据
# 参数estimator为基模型
# 参数n_features_to_select为选择的特征个数
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)

#带L1惩罚项的逻辑回归作为基模型的特征选择
SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)

#GBDT作为基模型的特征选择
SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)