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
    # 单变量图表
    # 1. 箱线图:属性和中位值的离散速度
    pyplot.show()
    # 2. 直方图:每个属性的分布情况
    dataset.hist()
    pyplot.show()
    # 多变量图表
    scatter_matrix(dataset)
    pyplot.show()
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

    # 6种算法模型 准确率比较
    models = {}
    models['LR'] = LogisticRegression()
    models['LDA'] = LinearDiscriminantAnalysis()
    models['KNN'] = KNeighborsClassifier()
    models['CART'] = DecisionTreeClassifier()
    models['NB'] = GaussianNB()
    models['SVM'] = SVC()

    results = []
    for key in models:
        kflod = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(models[key], X_train, Y_train, cv=kflod, scoring='accuracy')
        results.append(cv_results)
        print('%s: %f (%f)' % (key, cv_results.mean(), cv_results.std()))
    # 箱线图
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(models.keys())
    pyplot.show()
