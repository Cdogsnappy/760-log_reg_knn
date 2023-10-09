import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

k = [5]
#hyper-parameter

def knn(d, test):
    res_vals = []
    dists = getDists(d,test)
    num_labels = 0
    lowest_dist = sorted(dists, key=lambda tup: tup[0])
    for k_val in k:
        for val in lowest_dist[0:k_val]:
            num_labels+=val[1]
        if num_labels >= k_val/2:
            res_vals.append(1);
        else:
            res_vals.append(0);
        num_labels = 0
    return res_vals

def buildData(f1):
    d = []
    for line in f1:
        list1 = np.array([float(number) for number in line.split(' ')])
        d.append(list1)
    return d

def five_fold(d,):
    folds = [1000, 2000, 3000, 4000, 5000]
    end_res = []
    for f in folds:
        X_test = (d[f - 1000:f])
        X_train = np.append(d[0:f - 1000], d[f:len(d)], axis=0)
        tp = np.zeros(len(k))
        tn = np.zeros(len(k))
        fp = np.zeros(len(k))
        fn = np.zeros(len(k))
        for val in X_test:
            pred = knn(X_train, val)
            for ind in range(len(k)):
                if pred[ind] == val[len(val) - 1]:
                    if pred[ind] == 1:
                        tp[ind] += 1
                    else:
                        tn[ind] += 1
                else:
                    if pred[ind] == 1:
                        fp[ind] += 1
                    else:
                        fn[ind] += 1
        res = []
        for ind in range(len(k)):
            res.append(
                [k[ind], (tp[ind] + tn[ind]) / (tp[ind] + fp[ind] + tn[ind] + fn[ind]), tp[ind] / (tp[ind] + fn[ind]),
                 tp[ind] / (tp[ind] + fp[ind])])
        end_res.append([f, res])
    return end_res
def q4():
    f = open('emails.csv', 'r')
    d = buildEmails(f)
    end_res = five_fold(d)
    #print end_res
    avg_acc = []
    for k_val in range(len(k)):
        print("k = " + str(k[k_val]))
        acc = 0
        for val in end_res:
            #print("fold " + str(val[0]))
            #print("Accuracy: " + str(val[1][k_val][1]))
            #print("Recall: " + str(val[1][k_val][2]))
            #print("Precision: " + str(val[1][k_val][3]))
            #print("")
            acc+=val[1][k_val][1]
        avg_acc.append(acc/5)
        print("Average Accuracy: " + str(avg_acc[k_val]))
    plt.plot(k,avg_acc)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Average Accuracy vs. k")
    plt.show()



def q1():
    colors = ['red','blue']
    f = open('D2z.txt', 'r')
    d = np.array(buildData(f))
    for i in range(40):
        for j in range(40):
            plt.scatter(i/10-2,j/10-2,color=colors[knn(d,[i/10-2,j/10-2,0])[0]], s=5)
    plt.scatter(d[:,0],d[:,1],color='black')
    plt.show()

def buildEmails(f):
    X = []
    f.readline()
    for line in f:
        l = line.split(',')
        list1 = np.array([float(number) for number in l])
        X.append(list1)
    return np.array(X)

def getDists(data, test):
    diffs = np.square([test[0:len(test)-1] - d[0:len(d)-1] for d in data])
    dist = np.sqrt([np.sum(d) for d in diffs])
    return list(zip(dist, data[:, np.shape(data)[1]-1]))


def q5():
    f = open('emails.csv', 'r')
    preds = []
    points = []
    d = buildEmails(f)
    x_test = d[:1000]
    x_train = d[1000:]
    num_pos = np.sum(x_test[:, len(x_test[0]) - 1])
    num_neg = len(x_test) - num_pos
    for x in x_test:
        preds.append(knn_5(d,x)[0][1])
    preds = list(zip(x_test[:,len(x_test[0])-1], preds))
    preds = sorted(preds, key=lambda tup: tup[1])
    print(preds)
    tp = 0
    fp = 0
    last_tp = 0
    for i in range(len(x_test)):
        if i > 1 and preds[i][1] != preds[i - 1][1] and preds[i][0] == 0 and tp > last_tp:
            fpr = fp / num_neg
            tpr = tp / num_pos
            points.append([tpr, fpr])
            last_tp = tp
        if preds[i][0] == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / num_neg
        tpr = tp / num_pos
        points.append([tpr, fpr])
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1], color='red')

def knn_5(d, test):
    res_vals = []
    dists = getDists(d,test)
    num_labels = 0
    lowest_dist = sorted(dists, key=lambda tup: tup[0])
    for k_val in k:
        num_labels = 0
        for val in lowest_dist[0:k_val]:
            num_labels+=val[1]
        if num_labels >= k_val/2:
            res_vals.append([1,num_labels/k_val]);
        else:
            res_vals.append([0,num_labels/k_val]);
    return res_vals

def main():
    q1()

if __name__ == "__main__" :
    main()