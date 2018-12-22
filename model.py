import pickle

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

if __name__ == '__main__':
    # 导入数据集
    filename = 'data/iris.data'
    names = ['separ-length', 'separ-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(filename, names=names)
    # 概述数据
    print('数据维度:行 %s, 列 %s' % dataset.shape)
    print(dataset.head(10))
    print(dataset.describe())
    print(dataset.groupby('class').size())
    # 数据可视化
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    # 评估模型:80%数据训练，20%数据评估
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]
    validation_size = 0.2
    seed = 7
    # X_train, Y_train: 训练
    # X_validation, Y_validation： 评估验证
    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, test_size=validation_size, random_state=seed)
    svm_model = SVC()
    svm_model.fit(X=X_train, y=Y_train)
    predictions = svm_model.predict(X_validation)
    # print(accuracy_score(Y_validation, predictions))
    # print(confusion_matrix(Y_validation, predictions))
    # print(classification_report(Y_validation, predictions))
    with open('/Users/langshiquan/workspace/ml/iris/models/iris.model', 'wb') as f:
        pickle.dump(svm_model, f)
