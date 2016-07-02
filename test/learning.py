import pandas as pd


def loadDataSet():
    # train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
    train_url = 'train.csv'
    # test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
    test_url = 'test.csv'
    train = pd.read_csv(train_url)
    test = pd.read_csv(test_url)
    return train, test


def getSurvivedAndPassedAway(data):
    return data.Survived.value_counts()


def getPropertionsOfSurvivedAndPassedAway(data):
    return data.Survived.value_counts(normalize=True)


def getMalesWhoSurvived(data):
    return data.Survived[data.Sex == 'male'].value_counts()


def dealAgeWhichIsMissing(data):
    data['Age'] = data['Age'].fillna(data['Age'].median())


def createNewColumnChild(data):
    dealAgeWhichIsMissing(data)
    data['Child'] = float('NaN')
    data['Child'][data['Age'] < 18] = 1
    data['Child'][data['Age'] >= 18] = 0
    return data['Survived'][data['Child'] == 1].value_counts(normalize=True)


def describeData(data):
    return data.head(), data.describe()


def changeSexFromStringToInt(data):
    data['Sex'][data['Sex'] == 'male'] = 0
    data['Sex'][data['Sex'] == 'female'] = 1


def dealEmbarkedFormat(data):
    data['Embarked'] = data['Embarked'].fillna("S")  # deal missing value
    data['Embarked'][data['Embarked'] == 'S'] = 0
    data['Embarked'][data['Embarked'] == 'C'] = 1
    data['Embarked'][data['Embarked'] == 'Q'] = 2


def getFeaturesAndTarget(data):
    target = data['Survived'].values
    features = data[['Pclass', 'Sex', 'Age', 'Fare']].values
    return features, target


def getTree(features, target, maxDepth=10, minSimplesSplit=5):
    from sklearn import tree
    my_tree = tree.DecisionTreeClassifier(max_depth=maxDepth, min_samples_split=minSimplesSplit, random_state=1)
    return my_tree.fit(features, target)


def getTreeImportanceAndScore(tree, features, traget):
    importance = tree.feature_importances_
    score = tree.score(features, traget)
    return importance, score


def dealTestDataThatMissValueWithMedianOfColumn(data):
    data.Fare[152] = data['Fare'].median()


def getTestFeature(data):
    return data[['Pclass', 'Sex', 'Age', 'Fare']].values


def predictAndCreateDataFrame(tree, testFeatures, testData):
    import numpy as np
    prediction = tree.predict(testFeatures)
    passengerId = np.array(testData['PassengerId']).astype(int)
    solution = pd.DataFrame(prediction, passengerId, columns=['Survived'])
    solution.to_csv('solution.csv', index_label=['PassengerId'])
    return solution


def createNewColumnFamilySize(data):
    data['family_size'] = data.SibSp + data.Parch + 1


def test():
    train, testData = loadDataSet()
    # print(getSurvivedAndPassedAway(train))
    # print(getPropertionsOfSurvivedAndPassedAway(train))
    # print(getMalesWhoSurvived(train))
    createNewColumnChild(train)
    # print(describeData(train))
    changeSexFromStringToInt(train)
    dealEmbarkedFormat(train)
    features, target = getFeaturesAndTarget(train)
    tree = getTree(features, target)
    # print(getTreeImportanceAndScore(tree, features, target))

    dealTestDataThatMissValueWithMedianOfColumn(testData)
    changeSexFromStringToInt(testData)
    dealAgeWhichIsMissing(testData)
    testFeatures = getTestFeature(testData)
    solution = predictAndCreateDataFrame(tree, testFeatures, testData)
    print(solution)
