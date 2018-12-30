from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import numpy
import pandas

filename = 'WPBC.csv'

data = pandas.read_csv(filename,na_values = '?')
data = data.fillna(data.mean())
data = data[0::]  #Removed row 1 of  names
data = numpy.array(data) 

result = data[:,1]
Xnum = numpy.delete(data,(0,1),axis = 1) #Seperated other columns except OUTCOME and ID


from sklearn.model_selection import LeaveOneOut,train_test_split	#Cross Validation Leave one out method
loo = LeaveOneOut()
loo.get_n_splits(Xnum)

svclassifier =SVC(kernel = 'linear')
# i = 0

# for train_index,test_index in loo.split(Xnum):
	
# 	X_train,X_test = Xnum[train_index],Xnum[test_index]
# 	y_train,y_test = result[train_index],result[test_index]
X_train,X_test,y_train,y_test = train_test_split(Xnum,result,test_size =0.30) #default method of cross validation
# print('Loop:',i)
# i += 1
svclassifier.fit(X_train,y_train)
y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix  #Result and outcome
	# print(confusion_matrix(y_test,y_pred))
	# print(classification_report(y_test,y_pred))

print("Final Result")
y_pred = svclassifier.predict(Xnum)
print('Confusion Matrix',confusion_matrix(result,y_pred))
print('Report',classification_report(result,y_pred))