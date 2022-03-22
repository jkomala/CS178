
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.datasets import make_classification
import pandas as pd
import seaborn as sns
import mltools as ml


#Data Exploration
np.random.seed(0)#set seed

#Data set is split into 12 labels
red_data = np.genfromtxt("winequality-red.csv", delimiter =';')
white_data = np.genfromtxt("winequality-white.csv", delimiter =';')



red_y = red_data[:,-1] #target is last column
red_x = red_data[1:,0:-1] #features are other columns

white_y = white_data[:,-1]
white_x = white_data[1:,0:-1]

#Data Points:
print("RED:\nData points:", red_x.shape[0])
print("\nFeatures:", red_x.shape[1])

print("\nWHITE:\nData points:", white_x.shape[0])
print("\nFeatures:", white_x.shape[1])


#Creating histograms:

#Create figure with 11 subplots
fig, axs = plt.subplots(2,6, sharey=True, tight_layout = False, figsize = (15, 15))

#Have 11 features so might need to do it for all.

#RED
axs[0][0].hist(red_x[:,0])
axs[0][0].set_title("Red Fixed Acidity")
axs[0][1].hist(red_x[:,1])
axs[0][1].set_title("Red Volatile Acidity")
axs[0][2].hist(red_x[:,2])
axs[0][2].set_title("Red Citric Acid")
axs[0][3].hist(red_x[:,3])
axs[0][3].set_title("Red Residual sugar")
axs[0][4].hist(red_x[:,4])
axs[0][4].set_title("Red Chlorides")
axs[0][5].hist(red_x[:,5])
axs[0][5].set_title("Red Free sulfer dioxide")
axs[1][0].hist(red_x[:,6])
axs[1][0].set_title("Red Total sulfer dioxide")
axs[1][1].hist(red_x[:,7])
axs[1][1].set_title("Red Density")
axs[1][2].hist(red_x[:,8])
axs[1][2].set_title("Red pH")
axs[1][3].hist(red_x[:,9])
axs[1][3].set_title("Red Sulphates")
axs[1][4].hist(red_x[:,10])
axs[1][4].set_title("Red Alcohol")


plt.show()
plt.clf()



fig, axs = plt.subplots(2,6, sharey=True, tight_layout = False, figsize = (15, 15))

#WHITE
axs[0][0].hist(white_x[:,0])
axs[0][0].set_title("White Fixed Acidity")
axs[0][1].hist(white_x[:,1])
axs[0][1].set_title("White Volatile Acidity")
axs[0][2].hist(white_x[:,2])
axs[0][2].set_title("White Citric Acid")
axs[0][3].hist(white_x[:,3])
axs[0][3].set_title("White Residual sugar")
axs[0][4].hist(white_x[:,4])
axs[0][4].set_title("White Chlorides")
axs[0][5].hist(white_x[:,5])
axs[0][5].set_title("White Free sulfer dioxide")
axs[1][0].hist(white_x[:,6])
axs[1][0].set_title("White Total sulfer dioxide")
axs[1][1].hist(white_x[:,7])
axs[1][1].set_title("White Density")
axs[1][2].hist(white_x[:,8])
axs[1][2].set_title("White pH")
axs[1][3].hist(white_x[:,9])
axs[1][3].set_title("White Sulphates")
axs[1][4].hist(white_x[:,10])
axs[1][4].set_title("White Alcohol")


plt.show()
plt.clf()




#Random Forests.
#We have 11 labels
#data = pd.DataFrame({'Fixed Acidity':red_x.data[:,0],'Volatile Acidity': red_x.data[:,1],'Citric Acid' : red_x.data[:,2],'Residual Sugar': red_x.data[:,3],'Chlorides' : red_x.data[:,4],'Free Sulfer Dioxide': red_x.data[:,5],'Total Sulfur Dioxide' : red_x.data[:,6],'Density' : red_x.data[:,7],'pH' : red_x.data[:,8],'Sulphates' : red_x.data[:,9],'Alcohol' : red_x.data[:,10]'Quality' :red_x.target[:,11]})
X = red_data[1:,0:-1]
y = red_data[1:,-1]




#params = ['Fixed Acidity' 'Volatile Acidity', 'Citric Acid', 'Residual Sugar','Chlorides', 'Free Sulfer Dioxide', 'Total Sulfur Dioxide', 'Density', 'pH','Sulphates', 'Alcohol']
#df = pd.read_csv("winequality-red.csv")
#feature = red_y.data['Quality']


#Split 70/30 training test.
#split set into dependent and independent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#now we train the model and perform predictions
clf = RandomForestClassifier(n_estimators=100) #can change number of trees

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#Now we check accuracy
#How often is the classifier correct
print("Accuracy:", metrics.mean_squared_error(y_test, y_pred))


#visualize and adjust
feature_imp = pd.Series(clf.feature_importances_)#.sort_values(ascending=False)
print(feature_imp)


#Try it for white
X = white_data[1:,0:-1]
y = white_data[1:,-1]





#Split 70/30 training test.
#split set into dependent and independent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)



#now we train the model and perform predictions
clf = RandomForestClassifier(n_estimators=100) #can change.

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Now we check accuracy
print("Accuracy:", metrics.mean_squared_error(y_test, y_pred))

feature_imp = pd.Series(clf.feature_importances_)#.sort_values(ascending=False)
print(feature_imp)




# Split data
# Recall 1599 red. 1/5 = ~320
# 4898 white 1/5 = ~980

# Note, at some point we may want to combine red and white datasets, just keeping split for now to test on easier problem
ml.shuffleData(red_x, red_y)
ml.shuffleData(white_x, white_y)
red_valx = red_x[:320]
red_valy = red_y[:320]
white_valx = white_x[:980]
white_valy = white_y[:980]

red_trainx = red_x[321:]
red_trainy = red_y[321:]
white_trainx = white_x[981:]
white_trainy = white_y[981:]

tr_auc = np.zeros([])
val_auc = np.zeros([])

print('Data preparation complete')
print(red_valx.shape)
print(red_valy.shape)


# How to do feature selection?
# Originally planned to do something with entropy/information gain (remove features with particularly low IG)
# Realized our problem is regression-based, making that more complicated
# We'd need to set thresholds on the actual values, mimicking a decision tree (i.e. X1 < 4.6)

# Can we do another method of feature selection? 
# Reference: https://machinelearningmastery.com/feature-selection-for-regression-data/

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# Perform regression with only one feature and calculate correlation, repeat for all features, keep the top k features
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(red_trainx, red_trainy)

# yeah it's a long variable name, sue me.
selected_red_trainx = selector.transform(red_trainx)
scores = selector.scores_

for i in range(len(scores)):
    print(f'Feature {i+1} score: {scores[i]}')
    
print(selected_red_trainx[0]) # Verifies that ordering of features is intact
print(selected_red_trainx.shape)


# Test different layers/nodes
# mltools implementation

layers = [1, 2, 4, 6]
nodes = [5, 10, 25]    # Getting invalid value occurred in multiply/subtract errors for 1/2 layer 25 nodes (4/6 layers work?)

tr_auc = np.zeros((len(layers), len(nodes)))
va_auc = np.zeros((len(layers), len(nodes)))
for i,l in enumerate(layers):
    for j,n in enumerate(nodes):
        learner = ml.nnet.nnetRegress()
        size = [red_trainx.shape[1]]
        for val in range(l):
            size.append(n)
        size.append(1)
        
        learner.init_weights(size, 'random', red_trainx, red_trainy)
        learner.train(red_trainx, red_trainy, stopTol=1e-8, stepsize=0.25, stopIter=300)
        
        tr_auc[i,j] = learner.mse(red_trainx, red_trainy)
        va_auc[i,j] = learner.mse(red_valx, red_valy)
        
        print('Done Training/Evaluating AUC')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,5))
cax1 = ax1.matshow(tr_auc, interpolation='nearest')
cax2 = ax2.matshow(va_auc, interpolation='nearest')
f.colorbar(cax1, ax=ax1)
f.colorbar(cax2, ax=ax2)
ax1.set_xticklabels(['']+layers)
ax1.set_yticklabels(['']+nodes)
ax2.set_xticklabels(['']+layers)
ax2.set_yticklabels(['']+nodes)
ax1.set_title('Training')
ax2.set_title('Validation')
plt.show()


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(11, input_dim=red_trainx.shape[1], activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam')
print('Model compiled')


history = model.fit(red_trainx, red_trainy, epochs=100, batch_size=50, validation_split=0.2)

plt.clf()
plt.cla()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (MSE)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'])
plt.show()

