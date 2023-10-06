import numpy as np

class LogisticRegression():
    #hyper-parameters
    theta = 0
    iterations = 100
    alpha = .1

def gradient_descent(L,X,y):
    for i in range(L.iterations):
        y_hat = sigmoid(X.dot(L.theta))
        loss = np.reshape((y_hat - y.T),len(y))
        loss = np.dot(X.T,loss)
        L.theta = L.theta  - (L.alpha/len(y))*loss

def predict(L,v):
    z = sigmoid(v.dot(L.theta))
    y = np.where(z > 0.5, 1, 0)
    return y

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
        X_train = np.concatenate((X[0:f-1000], X[f:len(X)]),1)
        y_train = np.concatenate((y[0:f - 1000], y[f:len(y)]),1)
        L.theta = np.zeros(4000)
        gradient_descent(L,X_train,y_train)
        print(test_accuracy(L,X_test,y_test))

def test_accuracy(L,test_x,test_y):
    count = 0
    for val,y in test_x,test_y:
        pred = predict(L,val)
        if pred == y:
            count+=1
    return count/len(test_y)

def main():
    L = LogisticRegression()
    f = open('emails.csv', 'r')
    X, y = buildData(f)
    X = np.array(X)
    y = np.array(y)
    m,n = X.shape
    L.theta = np.zeros(n)
    gradient_descent(L,X,y)
    five_fold(L,X,y)


if __name__ == "__main__":
    main()