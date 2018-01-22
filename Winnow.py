
import numpy
# Perceptron Algorithm on the Sonar Dataset
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

def winnow(y, x, promotion_step):
    l = len(x[0])
    w=[1]* (l+1)
    #w = numpy.random.rand(l+1)                      # Initializing a random vector
    #w = w/float(numpy.linalg.norm(w))
    corrections = 0                                 # Variable to count the number of mistakes that would be made
    for count in range(len(x)):                  # For each record in the data set
        y_now = y[count]
        x_now = x[count]
        x_now_float = [ 1.0 ]
        for a in x_now:
            x_now_float.append(float(a))
        # print(len(w), len(numpy.cross(learning_rate*x_now_float)))
        #print(w,x_now_float)
        x_arr = numpy.array(x)
        avg_x= numpy.array(0)
        for z in range(6):
            print(sum(x_arr[: , z]) / len(x_arr[: , z]))
            avg_x[z] = sum(x_arr[:, z]) / len(x_arr[:, z])

        y_bar = numpy.dot(w, x_now_float)             # Find W^T*x_i
        x_now_float = numpy.asarray(x_now_float)
        if y_now != y_bar:
            corrections += 1        # Mistake, update the no of mistakes
            if y_bar == 0:                       # if (y_i)(W^T)(x_i) <= 0 ???
                for z in range(len(w)):
                    w[z] = w[z]*promotion_step
            else:
                for z in range(len(w)):
                    w[z] = w[z]*promotion_step
    return w, corrections

def validation(w, x_test, y_test):
    y_pred = []
    for b in x_test:
        x_test_fl = [1.0]
        for a in b:
            x_test_fl.append(float(a))
        x_test_fl = numpy.asarray(x_test_fl)
        y_here = numpy.dot(w, x_test_fl)
        if y_here > 0:
            y_pred.append(1)
        else:
            y_pred.append(0)
    correct = 0
    print(y_pred)
    for n in range(len(y_pred)):
        if y_pred[n] == y_test[n]:
            correct += 1
    print('Accuracy : ', correct/float(len(y_pred)))

def predict(w, x_eval, y_eval):
    y_eval = []
    for b in x_eval:
        x_eval_fl = [1.0]
        for a in b:
            x_eval_fl.append(float(a))
        x_eval_fl = numpy.asarray(x_eval_fl)
        y_here = numpy.dot(w, x_eval_fl)
        if y_here > 0:
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

#print(labels_eval, features_test[0])

step=2

[w, c] = winnow(labels_train, features_train, step)
print(w)
accuracy = validation(w, features_test, labels_test)

eval_label = predict(w, features_eval, labels_eval)

csv_array = []
for a in range(len(eval_label)):
    csv_array.append([index[a], eval_label[a]])

# write_csv('results.csv', )
write_csv('results.csv', csv_array)
# print(c, w)
