import numpy as np
import matplotlib.pyplot as plt
import knn

class LogisticRegression():
    #hyper-parameters
    theta = 0
    iterations = 300
    alpha = .03

def gradient_descent(L,X,y):
    for i in range(L.iterations):
        y_hat = sigmoid(X.dot(L.theta))
        loss = np.reshape((y_hat - y.T),len(y))
        loss = np.dot(X.T,loss)
        L.theta = L.theta  - (L.alpha/len(y))*loss

def predict(L,v):
    z = sigmoid(v.dot(L.theta))
    y = np.where(z > 0.5, 1, 0)
    return [y, z]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def buildData(f):
    X = []
    y = []
    f.readline()
    for line in f:
        l = line.split(',')
        list1 = [float(number) for number in l[1:len(l)-1]]
        X.append(list1)
        y.append(float(l[len(l)-1][0]))
    return X,y

def five_fold(L,X,y):
    folds = [1000,2000,3000,4000,5000]
    for f in folds:
        X_test = X[f-1000:f]
        y_test = y[f-1000:f]
        X_train = np.append(X[0:f-1000], X[f:len(X)], axis=0)
        y_train = np.append(y[0:f - 1000], y[f:len(y)], axis=0)
        L.theta = np.zeros(3001)
        gradient_descent(L,X_train,y_train)
        res = test_accuracy(L,X_test,y_test)
        print("fold " + str(f/1000))
        print("Accuracy: " + str(res[0]))
        print("Recall: " + str(res[1]))
        print("Precision: " + str(res[2]))
        print("")

def test_accuracy(L,test_x,test_y):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for val in range(len(test_x)):
        pred = predict(L,test_x[val])[0]
        if pred == test_y[val]:
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1
    return [(tp+tn)/(tp+tn+fp+fn), tp/(tp+fn), tp/(tp+fp)]

def q3():
    L = LogisticRegression()
    f = open('emails.csv', 'r')
    X, y = buildData(f)
    X = np.array(X)
    X = np.c_[np.ones(len(X)),X] #bias term
    y = np.array(y)
    m, n = X.shape
    L.theta = np.zeros(n)
    gradient_descent(L, X, y)
    five_fold(L, X, y)

def drawCurve(L,test_x,test_y):
    points = []
    num_pos = np.sum(test_y)
    num_neg = len(test_y)-num_pos
    print(num_pos)
    print(num_neg)
    preds = [predict(L,x)[1] for x in test_x]
    preds = list(zip(test_y,preds))
    preds = sorted(preds, key=lambda tup: tup[1])
    tp = 0
    fp = 0
    last_tp = 0
    for i in range(len(test_y)):
        if i > 1 and preds[i][1] != preds[i-1][1] and preds[i][0] == 0 and tp > last_tp:
            fpr = fp/num_neg
            tpr = tp/num_pos
            points.append([tpr,fpr])
            last_tp = tp
        if preds[i][0] == 1:
            tp+=1
        else:
            fp+=1
        fpr = fp / num_neg
        tpr = tp / num_pos
        points.append([tpr,fpr])
    points = np.array(points)
    plt.plot(points[:,0],points[:,1])
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.title("ROC Curve")

def q5():
    L = LogisticRegression()
    f = open('emails.csv', 'r')
    X, y = buildData(f)
    X = np.array(X)
    y = np.array(y)
    m, n = X.shape
    L.theta = np.zeros(n)
    X_test = X[0:1000]
    y_test = y[0:1000]
    X_train = X[1000:]
    y_train = y[1000:]
    gradient_descent(L, X_train, y_train)
    drawCurve(L,X_test,y_test)
    plt.show()
    knn.q5()
    plt.show()

def main():
    q5()


if __name__ == "__main__":
    main()