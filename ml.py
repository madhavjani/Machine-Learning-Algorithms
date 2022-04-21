import math
data = []
with open('breast-cancer-wisconsin.data', 'r') as f:
    indices = 0
    count = 0
    flag = False
    for line in f.readlines():
        count += 1
        attributes = line.strip('\n').split(',')
        if indices == 0:
            indices = len(attributes)
            labels = [0]*indices 
        if '?' not in attributes and flag == False:
            for i in range(0,len(attributes)):
                labels[i] += int(attributes[i])
        elif flag == True and '?' not in attributes:
            labels[index] += int(attributes[index])
        else:
            flag = True
            index = attributes.index('?')
       # print(attributes)
with open('breast-cancer-wisconsin.data', 'r') as f:
    for line in f.readlines():
        attributes = line.strip('\n').split(',')
        if '?' in attributes:
            attributes[index] = labels[index]/count
        data.append([int(x) for x in attributes])
print(data)

def info_dataset(data, verbose=True):
    label1, label2 = 0, 0
    data_size = len(data)
   
    for datum in data:
       
        if datum[-1] == 2:
            label1 += 1
        else:
            label2 += 1
    if verbose:
        print('Total of samples: %d' % data_size)
        print('Total label 1: %d' % label1)
        print('Total label 2: %d' % label2)
    return [len(data), label1, label2]

info_dataset(data)
p = 0.6
_, label1, label2 = info_dataset(data,False)

train_set, test_set = [], []
max_label1, max_label2 = int(p * label1), int(p * label2)
total_label1, total_label2 = 0, 0
for sample in data:
    if (total_label1 + total_label2) < (max_label1 + max_label2):
        train_set.append(sample)
        if sample[-1] == 2 and total_label1 < max_label1:
            total_label1 += 1
        else:
            total_label2 += 1
    else:
        test_set.append(sample)


def euclidian_dist(p1, p2):
    dim, sum_ = len(p1), 0
    for index in range(dim - 1):
        sum_ += math.pow(p1[index] - p2[index], 2)
    return math.sqrt(sum_)

def knn(train_set, new_sample, K):
    dists, train_size = {}, len(train_set)

    for i in range(train_size):
        d = euclidian_dist(train_set[i], new_sample)
        dists[i] = d

    k_neighbors = sorted(dists, key=dists.get)[:K]

    qty_label1, qty_label2 =0, 0
    for index in k_neighbors:
        if train_set[index][-1] == 2:
            qty_label1 += 1
        else:
            qty_label2 += 1

    if qty_label1 > qty_label2:
        return 1
    else:
        return 2

print(test_set[0])
print(knn(train_set, test_set[0], 12))

correct, K = 0, 15
for sample in test_set:
    label = knn(train_set, sample, K)
    if sample[-1] == label:
        correct += 1
print("Train set size: %d" % len(train_set))
print("Test set size: %d" % len(test_set))
print("Correct predicitons: %d" % correct)
print("Accunracy: %.2f%%" % (100 * correct / len(train_set)))