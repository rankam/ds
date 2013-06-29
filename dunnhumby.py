import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import csv
from sklearn.ensemble import GradientBoostingRegressor

# Read in file
csv_file_object = csv.reader(open('/users/aaronrank/datascience/kaggle/dunnhumby/train.csv', 'rb')) 
header = csv_file_object.next()                            
data=[]                          
for row in csv_file_object:      
    data.append(row)             
data = np.array(data)           

# Read in test file
csv_file_object = csv.reader(open('/users/aaronrank/datascience/kaggle/dunnhumby/test.csv', 'rb')) 
header = csv_file_object.next()                                  
testdata=[]                          
for row in csv_file_object:      
    testdata.append(row)
test_data = np.array(testdata) 

# Change nparray into pd DataFrame and remove unwanted data
train = pd.DataFrame(data)
train.pop(1)

# Add counter to be used for averaging weeks 10:13 data
# Averages will be used to replace missing datain weeks 14-26
idlist = range(1,(2768*26)+1)
train['count']=idlist
training=np.array(train.astype(np.float))

# Get averages for each column
averages=[]
for x in training[:,-1]:
	for y in range(3,17):
		if x % 13==0.0 and training[(x-1),1] != 26.0:
			averages.append((training[(x-2),y]+training[(x-3),y]+training[(x-4),y]+training[(x-5),y])/4)

# Convert list of averages into numpy array
averages = np.array(averages)
averages = np.reshape(averages,(2768,14))

# Combine numpy array of averages with existing data necessary for testing
combine_test = training[training[:,1]==26]
combine_test= combine_test[:,:3]
train_final_test =np.hstack((combine_test,averages))

# Create set of only weeks 1-13 to train on
training=training[training[:,1]<14.0]

# convert testing set to numpy array
test =np.array(test_data)
test[test[:,:]=='']=0
# Add counter to be used for averaging weeks 10:13 data
# Averages will be used to replace missing datain weeks 14-26
test = pd.DataFrame(test)
testidlist = range(1,28315)
# Change nparray into pd DataFrame and remove unwanted data while adding necessary data
test['count']=testidlist
test.pop(1)

# Get averages for each column
testing=np.array(test.astype(np.float))
testaverages=[]
for x in testing[:,-1]:
	for y in range(2,16):
		if x % 13==0.0 and testing[(x-1),1] != 26.0:
			testaverages.append((testing[x,y]+testing[(x-1),y]+testing[(x-2),y]+testing[(x-3),y])/4)
testaverages = np.array(testaverages)
testaverages = np.reshape(testaverages,(14,1089))
testaverages = np.transpose(testaverages)

combine = testing[testing[:,1]==26]
combine = combine[:,:3]
final_test =np.hstack((combine,testaverages))
# final_test = final_test[:,:15]

# log transform all data 
X_train= np.log(1+training[:,(1,2,4,5,6,7,8,9,10,11,12,13,14,15,16)])
y_train = np.log(1+training[:,3])
X_test = np.log(1+train_final_test[:,(1,2,4,5,6,7,8,9,10,11,12,13,14,15,16)])

# fit the data
clf = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1,max_depth=6, random_state=0, loss='ls').fit(X_train, y_train)
print 'cross validation', np.mean(cross_val_score(clf,X_train,y_train,cv=7))
print 'r^2',clf.score(X_train, y_train)
print 'Feature Importance', clf.feature_importances_
prediction = clf.predict(X_test)
for predictions in prediction:
	print predictions
# print sklearn.metrics.r2_score(np.log(1+train_final_test[:,3]), prediction)


