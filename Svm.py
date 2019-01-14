from sklearn.svm import SVC
import numpy
import pandas
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.model_selection import LeaveOneOut	#Cross Validation Leave one out method
loo = LeaveOneOut()
from sklearn.metrics import classification_report,confusion_matrix  #Result and outcome

def topDownApproach(Xnum):
	Xnum = numpy.array(Xnum)
	Xnum = Xnum.T

	loo.get_n_splits(Xnum)

	svclassifier =SVC(kernel = 'linear')
	i = 0

	for train_index,test_index in loo.split(Xnum):
		
		X_train = Xnum[train_index]
		y_train = result[train_index]
	# X_train,X_test,y_train,y_test = train_test_split(Xnum,result,test_size =0.30) #default method of cross validation
		print('Loop:',i)
		i += 1
		svclassifier.fit(X_train,y_train)

	print("Final Result")
	y_pred = svclassifier.predict(Xnum)
	print('Confusion Matrix',confusion_matrix(result,y_pred))
	print('Report',classification_report(result,y_pred))

filename = 'WPBC.csv'

data = pandas.read_csv(filename,na_values = '?')
data = data.fillna(data.mean())
# data = data[0::]  #Removed row 1 of  names
data = numpy.array(data)
# data = numpy.delete(data,(0,1),axis = 1)
data = data[:,[1,2,5,6,15,16,25,26,33,34]]
result = data[:,0]
Xnum = []
data = numpy.delete(data,0,axis = 1)  

for i in range(9):
	Xnum.append(data[:,i])
	topDownApproach(Xnum)


