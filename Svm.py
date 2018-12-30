from sklearn.svm import SVC
import numpy
import pandas

filename = 'WPBC.csv'
# raw_data = open(filename,'rt')
# reader = csv.reader(raw_data,delimiter = ',', quoting = csv.QUOTE_NONE)

data = pandas.read_csv(filename,na_values = '?')
data = data.fillna(data.mean())
data = data[0::]  #Removed column names

Xnum = numpy.array(data) 
result = Xnum[:,0] #Seperated outcome column
Xnum = numpy.delete(Xnum,(0,1),axis = 1) #Seperated other columns except OUTCOME and ID



from sklearn.model_selection import LeaveOneOut	#Cross Validation Leave one out method
loo = LeaveOneOut()
loo.get_n_splits(Xnum)
i = 0
for train_index,test_index in loo.split(Xnum):
	X_train,X_test = Xnum[train_index],Xnum[test_index]
	y_train,y_test = result[train_index],result[test_index]
	# X_train,X_test,y_train,y_test = train_test_split(Xnum,result,test_size =0.20) #default method of cross validation
	print('Loop:',i)
	i+=1
	svclassifier = SVC(kernel = 'linear')
	print(X_train,y_train)
	svclassifier.fit(X_train,y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metric import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))