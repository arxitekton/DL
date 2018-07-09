
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler


train_filename = "adult.data"
test_filename = "adult.test"

column_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','target']

train_data = pd.read_csv(train_filename, names=column_names, skiprows=1, header=None)
test_data = pd.read_csv(test_filename, names=column_names, skiprows=1, header=None)

splitIdx = train_data.shape[0]

data = pd.concat([train_data, test_data], ignore_index=True)

# CATEGORICAL
catFeatures = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']

# Convert categorical variable into dummy/indicator variables
df_cat = pd.get_dummies(data[catFeatures])

# concat dummy/indicator variables# concat
data = pd.concat([data, df_cat], axis=1)

data['target'] = data['target'].map({' <=50K.': ' <=50K', ' >50K.': ' >50K'})

# drop useless categorical column
data.drop(catFeatures, axis=1,inplace=True)

#INTEGER
intFeatures = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']

# integer features scaling
scaler = MinMaxScaler()
data[intFeatures] = scaler.fit_transform(data[intFeatures])

# split features and label(class)
dfX = data.drop(['target'], axis=1)
dfY = pd.get_dummies(data['target'])

# get features names
fNames = dfX.columns

# get target names
tNames = dfY.columns

features = dfX.values
label = dfY.values

biasPad = np.ones((features.shape[0],1), dtype=features.dtype)
features = np.concatenate((features,biasPad), axis=1)

XTest = features[splitIdx:,:]
XTrain = features[:splitIdx,:]

YTest = label[splitIdx:,:]
YTrain = label[:splitIdx,:]



def softmax(x):
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    f = np.exp(shifted_x)
    p = f / np.sum(f, axis=1, keepdims=True)

    return p

def relu(x):
    result = np.maximum(0.0, x)

    return result

def d_relu(a):
    d = np.zeros_like(a)
    d[np.where(a >= 0.0)] = 1.0

    return d

def tanh(x):

    return np.tanh(x)

def d_tanh(x):

    return 1.0 - np.tanh(x) ** 2


def cross_entropy(p, t):
    s = t.shape[0]
    log_likelihood = -np.log(p[range(s), np.argmax(t, axis=1)])

    return np.sum(log_likelihood) / s


def d_cross_entropy(g, y):
    m = y.shape[0]
    g[range(m), np.argmax(y, axis=1)] -= 1
    g = g / m

    return g


class NN:
    def __init__(self, ni, nh1, nh2, no):
        self.ni = ni
        self.nh1 = nh1
        self.nh2 = nh2
        self.no = no

        self.wih1 = np.random.rand(ni, nh1) / (np.sqrt(ni * nh1) * 1e3)
        self.wh1h2 = np.random.rand(nh1, nh2) / (np.sqrt(nh1 * nh2) * 1e3)
        self.wres = np.random.rand(ni, no) / (np.sqrt(ni * no) * 1e3)
        self.wout = np.random.rand(nh2, no) / (np.sqrt(nh2* no) * 1e3)

        self.bh1 = np.random.rand(nh1) / (np.sqrt(nh1) * 1e3)
        self.bh2 = np.random.rand(nh2) / (np.sqrt(nh2) * 1e3)
        self.bo = np.random.rand(no) / (np.sqrt(no) * 1e3)

    def cross_entropy(self, p, t):
        s = t.shape[0]
        log_likelihood = -np.log(p[range(s), np.argmax(t, axis=1)])

        return np.sum(log_likelihood) / s

    def d_cross_entropy(self, g, y):
        m = y.shape[0]
        g[range(m), np.argmax(y, axis=1)] -= 1
        g = g / m

        return g

    def feedFwd(self, X):
        ai = X
        ah1 = tanh(np.dot(ai, self.wih1) + self.bh1)
        ah2 = relu(np.dot(ah1, self.wh1h2) + self.bh2)
        ao = softmax(np.dot(ah2, self.wout) + np.dot(ai, self.wres) + self.bo)

        return ai, ah1, ah2, ao


    def backProp(self, ai, ah1, ah2, ao, y, batchSize=1):

        delOut = self.d_cross_entropy(ao, y)
        delHidden2 = delOut.dot(self.wout.T) * d_relu(ah2)
        delHidden1 = delHidden2.dot(self.wh1h2.T) * d_tanh(ah1)

        self.wout -= self.learning_rate * ah2.T.dot(delOut)
        self.wres -= self.learning_rate * ai.T.dot(delOut)
        self.wh1h2 -= self.learning_rate * ah1.T.dot(delHidden2)
        self.wih1 -= self.learning_rate * ai.T.dot(delHidden1)

        self.bo -= self.learning_rate * np.sum(delOut, axis=0)
        self.bh2 -= self.learning_rate * np.sum(delHidden2, axis=0)
        self.bh1 -= self.learning_rate * np.sum(delHidden1, axis=0)

    def fit(self, X, y, alpha=0.1):
        self.learning_rate = alpha
        ai, ah1, ah2, ao = self.feedFwd(X)
        self.out_error = self.cross_entropy(ao, y)
        self.backProp(ai, ah1, ah2, ao, y)

    def predict(self, X):
        ai, ah1, ah2, ao = self.feedFwd(X)
        return ao

def accuracy(pred, y_true):
    pr_cl = [np.argmax(pred) + 1 for pred in pred]
    y_cl = [np.argmax(pred) + 1 for pred in y_true]
    res = 0
    for xc, yc in zip(pr_cl, y_cl):
        res += 0 if xc == yc else 1
    return ((len(pr_cl) - res)/len(pr_cl))*100

def evaluate(model, X_train, y_train, X_test, y_test, iteration=100, alpha = 0.1, batch_size = 10):

    epochList = []
    accur = []
    error = []

    batch_count = int(np.ceil(len(X_train)/batch_size))
    for epoch in range(iteration):
        for batch_number in range(batch_count):
            batch_offset = batch_number*batch_size
            batch_X = X_train[batch_offset:min(batch_offset+batch_size, X_train.shape[0]),:]
            batch_y = y_train[batch_offset:min(batch_offset+batch_size, y_train.shape[0]),:]
            model.fit(batch_X, batch_y, alpha=alpha)
        y_pred = model.predict(X_test)
        acc = accuracy(y_pred, y_test)
        accur.append(acc)
        err = np.mean(np.abs(model.out_error))
        error.append(err)

    return epochList, error, acc


model = NN(ni=XTrain.shape[1], nh1=55, nh2=55, no = YTrain.shape[1])

iteration = 100
alpha = 0.005
batch_size = 40


epochList, error_list, accuracy_list = evaluate(model, XTrain, YTrain, XTest, YTest, iteration, alpha, batch_size)

#plot graph
plt.plot(epochList, error_list)
plt.plot(epochList, accuracy_list)
plt.xlabel('Number of epoch')
plt.ylabel('Error')
plt.savefig('loss-function.png')
plt.show()