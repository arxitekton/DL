
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


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
data.drop(catFeatures, axis=1, inplace=True)

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
    d[np.where(a > 0.0)] = 1.0

    return d

def tanh(x):

    return np.tanh(x)

def d_tanh(x):

    return 1.0 - x ** 2

# https://deepnotes.io/softmax-crossentropy
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

        # He Initialization
        # self.wih1 = np.random.rand(inputCount, hiddenCount1)*np.sqrt(2./inputCount)

        # Xavier Parameter Initialization
        self.wih1 = np.random.rand(ni, nh1) * np.sqrt(6) / (np.sqrt(ni * nh1))
        self.wh1h2 = np.random.rand(nh1, nh2) * np.sqrt(6) / (np.sqrt(nh1 * nh2))
        self.wres = np.random.rand(ni, no) * np.sqrt(6) / (np.sqrt(ni * no))
        self.wout = np.random.rand(nh2, no) * np.sqrt(6) / (np.sqrt(nh2* no))

        self.bh1 = np.random.rand(nh1) * np.sqrt(6) / (np.sqrt(nh1))
        self.bh2 = np.random.rand(nh2) * np.sqrt(6) / (np.sqrt(nh2))
        self.bo = np.random.rand(no) * np.sqrt(6) / (np.sqrt(no))
        self.alpha = 0.01


    def feedFwd(self, features):
        ai = features
        ah1 = tanh(np.dot(ai, self.wih1) + self.bh1)
        ah2 = relu(np.dot(ah1, self.wh1h2) + self.bh2)
        ao = softmax(np.dot(ah2, self.wout) + np.dot(ai, self.wres) + self.bo)

        return ai, ah1, ah2, ao


    def backProp(self, ai, ah1, ah2, ao, y, batchSize=1):

        delOut = d_cross_entropy(ao, y)
        delHidden2 = delOut.dot(self.wout.T) * d_relu(ah2)
        delHidden1 = delHidden2.dot(self.wh1h2.T) * d_tanh(ah1)

        self.wout = self.alpha * ah2.T.dot(delOut)/batchSize
        self.wres = self.alpha * ai.T.dot(delOut)/batchSize
        self.wh1h2 = self.alpha * ah1.T.dot(delHidden2)/batchSize
        self.wih1 = self.alpha * ai.T.dot(delHidden1)/batchSize

        self.bo = self.alpha * np.sum(delOut, axis=0)/batchSize
        self.bh2 = self.alpha * np.sum(delHidden2, axis=0)/batchSize
        self.bh1 = self.alpha * np.sum(delHidden1, axis=0)/batchSize

    def fit(self, X, y, batchSize=1, alpha=0.1):
        self.alpha = alpha
        ai, ah1, ah2, ao = self.feedFwd(X)
        self.out_error = cross_entropy(ao, y)
        self.backProp(ai, ah1, ah2, ao, y, batchSize=1)

    def predict(self, X):
        ai, ah1, ah2, ao = self.feedFwd(X)
        return ao


def accuracy_metric(actual, predicted):
    actual = np.argmax(actual, axis=1)
    predicted = np.argmax(predicted, axis=1)

    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) * 100.0


def train(model, X, Y, iteration=1000, alpha=0.001, batchSize=1, beta=0.099, decayRate=0.0005):
    errorTimeline = []
    epochList = []

    # train it for iteration number of epoch
    for epoch in range(iteration):

        # for each mini batch
        for i in range(0, X.shape[0], batchSize):
            # split the dataset into mini batches
            batchSplit = min(i + batchSize, X.shape[0])
            XminiBatch = X[i:batchSplit, :]
            YminiBatch = Y[i:batchSplit]

            # calculate a forwasd pass through the network
            ai, ah1, ah2, ao = model.feedFwd(XminiBatch)

            # calculate mean squared error
            error = 0.5 * np.sum((YminiBatch - ao) ** 2) / batchSize
            # print error

            # backprop and update weights
            model.backProp(ai, ah1, ah2, ao, YminiBatch, batchSize=1)

        # after every 50 iteration decrease momentum and learning rate
        # decreasing momentum helps reduce the chances of overshooting a convergence point
        step = 1
        if epoch % step == 0 and epoch > 0:
            model.alpha *= 1. / (1. + (decayRate * epoch))
            #beta *= 1. / (1. + (decayRate * epoch))
            # Store error for ploting graph
            errorTimeline.append(error)
            epochList.append(epoch)
            print('Epoch :', epoch, ', Error :', error, ', alpha :', model.alpha)

    return epochList, errorTimeline

model = NN(ni=XTrain.shape[1], nh1=35, nh2=35, no=YTrain.shape[1])

iteration = 100
alpha = 0.01
batch_size = 20


epochList, error_list = train(model, XTrain, YTrain, iteration=iteration, alpha=alpha, batchSize=batch_size, beta=0.099, decayRate=0.0005)


#plot graph
plt.plot(epochList, error_list)
plt.xlabel('Number of epoch')
plt.ylabel('Error')
plt.savefig('loss-function.png')
plt.show()

