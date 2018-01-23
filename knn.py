import numpy
# K Nearest Neighbours
from random import seed
from random import randrange
from csv import reader, writer
import time
from multiprocessing.dummy import Pool as ThreadPool



# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
        #print(dataset)
    return dataset

def write_csv(filename, array):
    with open(filename, 'w') as csvfile:
        csv_writer = writer(csvfile)
        # [['Id', 'Prediction']]
        csvfile.write('Id,Prediction\n')
        for row in array:
            csvfile.write(str(row[0][0]) + ',' + str(row[1]) + '\n')

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    new_a = []
    new_b = []
    c = []

    for i in range(len(b)):
        c.append(i)

    numpy.random.shuffle(c)

    for i in c:
        new_a.append(a[i])
        new_b.append(b[i])

    return new_a, new_b

def sarath(i, x, y):
    for a in i:
        new = []
        new.append(a.split(" "))
        if new[0][0] == '0':
            y.append(0)
        elif new[0][0] == '1':
            y.append(1)
        row = []
        for l in range(1, len(new[0])):
            [c, b] = new[0][l].split(":")
            row.append(b)
        x.append(row)
    return x, y

def sarath_test(i, x):
    for a in i:
        new = []
        new.append(a.split(" "))
        row = []
        for l in range(len(new[0])):
            [c, b] = new[0][l].split(":")
            row.append(b)
        x.append(row)
    return x

def cal_dist(x_train, y_train, x_test_h_f, iter_n):
    x_train_h_f = []


def knn(x_train, y_train, x_test, num_of_neigh):
    y_pred = []
    # [x_train, y_train] = unison_shuffled_copies(x_train, y_train)
    for test_iter in range(len(x_test)):
        if test_iter%100 == 0:
            print('currently working on :', test_iter)

        x_test_h_f = []
        for a in x_test[test_iter]:
            x_test_h_f.append(float(a))
        x_test_h_f = numpy.asarray(x_test_h_f)
        dist = []

        # for iter_n in range(len(x_train)):
        #     dist.append(cal_dist(x_train, y_train, x_test_h_f, iter_n))

        for i in range(len(x_train)):
            x_train_h_f = []
            for b in x_train[i]:
                x_train_h_f.append(float(b))
            x_train_h_f = numpy.asarray(x_train_h_f)
            dist.append([numpy.linalg.norm(x_test_h_f-x_train_h_f), y_train[i]])

        # sorted(dist)
        dist.sort()
        dist = dist[0:num_of_neigh]

        labels = numpy.transpose(dist)[1]
        if sum(labels) > num_of_neigh/2:
            y_pred.append(1)
        else:
            y_pred.append(0)

    assert len(y_pred) == len(x_test)
    return y_pred

def validate_knn(y_pred, y_test):
    correct_predictions = 0
    print(len(y_pred), len(y_test))
    assert len(y_pred) == len(y_test)
    total_predictions = len(y_pred)
    for i in range(total_predictions):
        if y_pred[i] == y_test[i]:
            correct_predictions += 1

    correct_predictions = (correct_predictions / float(total_predictions)) * 100
    return correct_predictions


def predict(w, x_eval):
    y_eval = []
    for b in x_eval:
        x_eval_fl = [1.0]
        for a in b:
            x_eval_fl.append(float(a))
        x_eval_fl = numpy.asarray(x_eval_fl)
        y_here = numpy.dot(w, x_eval_fl)
        if y_here >= 0:
            y_eval.append(1)
        else:
            y_eval.append(0)
    return y_eval


seed(1)
start_time = time.time()
# load and prepare data
filename = 'data.train'
dataset_train = load_csv(filename)

filename = 'data.test'
dataset_test = load_csv(filename)

filename = 'data.eval.anon'
dataset_eval = load_csv(filename)

filename = 'data.eval.id'
index = load_csv(filename)
# print(index)

features_train = []
labels_train = []

features_test = []
labels_test = []

features_eval =[]
labels_eval = []

for i in dataset_train:
    [features_train, labels_train] = sarath(i, features_train, labels_train)

for i in dataset_test:
    [features_test, labels_test] = sarath(i, features_test, labels_test)

for i in dataset_eval:
    [features_eval, labels_eval] = sarath(i, features_eval, labels_eval)

accuracies =[]
weights = []

predictrions = knn(features_train, labels_train, features_test, 13)
accuracies.append(validate_knn(predictrions, labels_test))
# accuracies.append(validate(w, features_test, labels_test))
# weights.append(w)

print(accuracies)


print("--- %s seconds ---" % (time.time() - start_time))

exit(1)
o = numpy.argmax(accuracies)
print('o', o, len(weights), len(accuracies))
print(weights[o])

eval_label = predict(weights[o], features_eval)

csv_array = []
for a in range(len(eval_label)):
    csv_array.append([index[a], eval_label[a]])

# write_csv('results.csv', )
write_csv('results.csv', csv_array)
# print(c, w)
