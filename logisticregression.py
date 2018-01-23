import numpy
# Logistic Regression Algorithm on the Twitter Data Set
from random import seed
from random import randrange
from csv import reader, writer

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


def svm(y, x, gamma0, c):
    print("learning_rate = ", gamma0, " C = ", c)
    # print(x)
    l = len(x[0])
    w = numpy.zeros(l+1)
    for each_epochs in range(5):
        [x, y] = unison_shuffled_copies(x, y)
        for count in range(len(x)):                  # For each record in the data set
            y_now = y[count]
            if y_now == 0:
                y_now = -1
            x_now = x[count]
            x_now_float = [ 1.0 ]
            for a in x_now:
                x_now_float.append(float(a))
            x_now_float = numpy.asarray(x_now_float)

            gamma_t = gamma0 / float(1 + gamma0 * count / c)
            product = y_now * numpy.dot(w, x_now_float)
            if product <= 1:
                w = (1 - gamma_t) * w + gamma_t * c * y_now * x_now_float
            else:
                w *= (1 - gamma_t)
            # learning_rate_t = learning_rate / float(1 + learning_rate * count /c)
            # product = y_now * numpy.dot(w, x_now_float)
            # gradient = (y_now * x_now_float) / float(numpy.exp(product) + 1) - 2 * w / float(c * c)
            # w += learning_rate_t * gradient
    return w


def Logistic_regression(y, x, gamma0, sigma):
    print("learning_rate = ", gamma0, 'Sigma : ', sigma)
    # print(x)
    l = len(x[0])
    w = numpy.zeros(l+1)
    for each_epochs in range(5):
        [x, y] = unison_shuffled_copies(x, y)
        for count in range(len(x)):                  # For each record in the data set
            y_now = y[count]
            if y_now == 0:
                y_now = -1
            x_now = x[count]
            x_now_float = [ 1.0 ]
            for a in x_now:
                x_now_float.append(float(a))
            x_now_float = numpy.asarray(x_now_float)

            gamma_t = gamma0 / float(1 + gamma0 * count / sigma)
            product = y_now * numpy.dot(w, x_now_float)
            gradient = (y_now * x_now_float) / float(numpy.exp(product) + 1) - 2 * w / float(sigma * sigma)
            w += gamma_t * gradient

            # learning_rate_t = learning_rate / float(1 + learning_rate * count /c)
            # product = y_now * numpy.dot(w, x_now_float)
            # gradient = (y_now * x_now_float) / float(numpy.exp(product) + 1) - 2 * w / float(c * c)
            # w += learning_rate_t * gradient
    return w


def validate(w, x_test, y_test):
    correct_predictions = 0                             # Variable to save the number of corrections
    tp = 0                                              # True Positive variable
    fp = 0                                              # False Positive variable
    fn = 0                                              # False Negative variable
    total_predictions = len(x_test)                 # Number of records ?
    for current_prediction in range(total_predictions): # loop over all the records
        y_here = y_test[current_prediction]
        x_here = x_test[current_prediction]
        if y_here == 0:
            # print('Yes!')
            y_here = -1
        x_test_fl = [1.0]
        for a in x_here:
            x_test_fl.append(float(a))
        x_test_fl = numpy.asarray(x_test_fl)
        predicted = y_here * numpy.dot(w, x_test_fl)     # What is the prediction of W (final vector) ?
        if predicted >= 0:
            correct_predictions += 1
        # print('y_here - ', y_here, ' y_pred - ', numpy.dot(w, x_test_fl))
        if y_here >=0 and numpy.dot(w, x_test_fl) >= 0:
            tp += 1
        elif y_here <0 and numpy.dot(w, x_test_fl) >= 0:
            fp += 1
        elif y_here >=0 and numpy.dot(w, x_test_fl) < 0:
            fn += 1
    correct_predictions = (correct_predictions/float(total_predictions))*100
    # print('------', correct_predictions, tp, fp, fn)
    return correct_predictions #, tp, fp, fn


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
for l_rate in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]:
    for c in [100, 50, 10, 1, 0.5, 0.1, 0.05, 0.01, 0.005]:
        w = Logistic_regression(labels_train, features_train, l_rate,c)
        accuracies.append(validate(w, features_test, labels_test))
        weights.append(w)

print(accuracies)

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
