import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

if __name__ == "__main__":

    df = pd.read_csv("Hotels_data_Changed.csv")
    newdf = df[["WeekDay","Snapshot Date","Checkin Date","DayDiff","Hotel Name","Discount Code"]]
    print(newdf.columns)
    print(newdf.head(5))
    le1 = preprocessing.LabelEncoder()
    le1.fit(newdf['Hotel Name'])
    newdf['Hotel Name'] = le1.transform(newdf['Hotel Name'])
    le2 = preprocessing.LabelEncoder()
    le2.fit(newdf['Snapshot Date'])
    newdf['Snapshot Date'] = le2.transform(newdf['Snapshot Date'])
    le3 = preprocessing.LabelEncoder()
    le3.fit(newdf['Checkin Date'])
    newdf['Checkin Date'] = le3.transform(newdf['Checkin Date'])
    le4 = preprocessing.LabelEncoder()
    print(newdf.head(5))
    train_df = newdf.head(int(len(newdf)*0.3))
    train_x = train_df[["WeekDay","Snapshot Date","Checkin Date","DayDiff","Hotel Name"]] ## Features
    train_y = train_df["Discount Code"] ## Target class
    ##clf = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth = 3, min_samples_leaf = 5)
    ##clf = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 5, min_samples_leaf = 5)
    clf = tree.DecisionTreeClassifier()

    clf = clf.fit(train_x, train_y)
    ##print(clf.predict([[2,0,32,33.0,35]]))
    test_df = newdf.tail(int(len(newdf) * 0.1))
    ##
    test_x = test_df[["WeekDay", "Snapshot Date", "Checkin Date", "DayDiff", "Hotel Name"]]  ## Features
    test_y = test_df["Discount Code"].tolist()  ## Target class
    print(test_x)
    predictions = clf.predict(test_x)
    print(clf.predict(test_x))
    hitcount = 0
    for index in range(len(test_y)):
        res1 = test_y[index]
        res2 = predictions[index]
        if res1 == res2:
            hitcount = hitcount + 1
    print("Number of hits :" + str(hitcount/len(test_y)))

## Statistics calculation

    print("MSE is : " + str(mean_squared_error(test_y, predictions)))

    confusion_matrix(test_y, predictions)
    ##tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    print(test_y)
    print(predictions.tolist())

    returned = confusion_matrix(test_y, predictions.tolist())##.ravel()
    print(returned)

    fpr, tpr, thresholds = roc_curve(test_y, predictions.tolist(), pos_label=2)
    print(fpr)
    print(tpr)
    print(thresholds)

    roc_auc = auc(fpr, tpr)

    plt.title('Decision Tree Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

##

    gnb = GaussianNB()
    gnb.fit(train_x, train_y)
    y_pred = gnb.predict(test_x)
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(test_y, y_pred) * 100)

    ## Statistics calculation

    print("MSE is : " + str(mean_squared_error(test_y, y_pred)))

    confusion_matrix(test_y, y_pred)
    print(test_y)
    print(y_pred.tolist())

    returned = confusion_matrix(test_y, y_pred.tolist())  ##.ravel()
    print(returned)

    fpr, tpr, thresholds = roc_curve(test_y, y_pred.tolist(), pos_label=2)
    print(fpr)
    print(tpr)
    print(thresholds)

    roc_auc = auc(fpr, tpr)

    plt.title('Gaussian Naive Base Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

##

