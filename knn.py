import numpy as np

k = 1
#hyper-parameter

def knn(d, test):
    dists = []
    num_labels = 0
    for i in d:
        dist.append(getDist(i,test))
    lowest_dist = sorted(dist, key=lambda tup: tup[0])
    for val in lowest_dist[0:k]:
        num_labels+=val[1]
    if num_labels > k/2:
        return 1;
    return 0;

def buildData(f1):
    d = []
    for line in f1:
        list1 = [float(number) for number in line.split(' ')]
        d.append(list1)
    return d

def getDist(p, test):
    dist = 0
    for i in range(len(p)-2):
        dist+=(test[i] - p[i])**2
    dist = np.sqrt(dist)
    return [dist,p[len(p)-1]]



def main():
    f = open('D2z.txt', 'r')
    d = buildData(f)



if __name__ == "__main__" :
    main()