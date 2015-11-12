import pandas as pd
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":

    data_set = pd.read_csv('data/train.csv')
    train = data_set.sample(frac=0.66)
    test = data_set.drop(train.index.values.tolist())

    X_train = train.iloc[:, 1:].values
    Y_train = train[[0]].values.ravel()
    X_test = test.iloc[:, 1:].values
    Y_test = test[[0]].values.ravel()

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, Y_train)
    rf.predict(X_test)

    # Roughly 96% accuracy
    print('Overall Accuracy: {0:3f}%'.format(rf.score(X_test, Y_test) * 100))
