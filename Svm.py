from sklearn.svm import SVC
import Database 
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut	#Cross Validation Leave one out method
from sklearn.metrics import classification_report,confusion_matrix  #Result and outcome
from sklearn.model_selection import train_test_split,KFold

loo = LeaveOneOut()

def topDownApproach(Head_Data):

	loo.get_n_splits(Head_Data)

	svclassifier =SVC(kernel = 'rbf',gamma = 'auto')
	
	i = 0
	Head_Data = np.array(Head_Data)
	Dataframe.Result = np.array(Dataframe.Result)

	for train_index,test_index in loo.split(Head_Data):
		
		X_train,y_train = Head_Data[train_index] , Dataframe.Result[train_index]

		print('Loop:',i)
		i += 1
		svclassifier.fit(X_train,y_train)

	print("Final Result")
	y_pred = svclassifier.predict(Head_Data)
	print('Confusion Matrix',confusion_matrix(Dataframe.Result,y_pred))
	print('Report',classification_report(Dataframe.Result,y_pred)) # Check if you could use Guassian kernel



def randomizeSplit(Dataset):
	x_train, x_test, y_train, y_test = train_test_split(Dataset.Data,Dataset.Result, test_size=.2)
	svclassifier = SVC(kernel = 'rbf',C = 1,gamma = 'auto')
	svclassifier.fit(x_train,y_train)
	y_pred = svclassifier.predict(x_test)	
	print("Confusion Matrix",confusion_matrix(y_test,y_pred))
	print('Report',classification_report(y_test,y_pred))


def kFoldCrossValidation(Dataset):
	svclassifier = SVC(kernel = 'rbf',C = 1,gamma = 'auto')
	kfold = KFold(n_splits = 5,shuffle = True,random_state = 42) # Split in 10 folds,shuffle prior to split true,value for pseudorandom number generator
	Dataframe.Data = np.array(Dataframe.Data)
	Dataframe.Result = np.array(Dataframe.Result)
	for train,test in kfold.split(Dataset.Data):
		print(Dataset.Data)
		print('Train: ',train,' Test: ',test)
		x_train, x_test, y_train, y_test = Dataset.Data[train], Dataset.Data[test], Dataset.Result[train], Dataset.Result[test]	
		svclassifier.fit(x_train,y_train)
		y_pred = svclassifier.predict(x_test)	
		print("Confusion Matrix",confusion_matrix(y_test,y_pred))
		print('Report',classification_report(y_test,y_pred))


# Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. 
# 			The question that this metric answer is of all passengers that labeled as survived, how many actually survived? 
# 			High precision relates to the low false positive rate. Higher the value of precision better it is.
# Precision = TP/TP+FP

# Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. 
# 					The question recall answers is: Of all the passengers that truly survived, how many did we label? 
# 					Recall above 0.5 is good and more higher the value better it is.
# Recall = TP/TP+FN

# F1 score - F1 Score is the weighted average of Precision and Recall. Therefore,
# 		This score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, 
# 		but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. 
# 		Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, 
# 		it’s better to look at both Precision and Recall.

# F1 Score = 2*(Recall * Precision) / (Recall + Precision)



Dataframe = Database.Dataset()
Dataframe.df = pd.read_csv('F:\\Coding\\Machine Learning\\Research-Paper\\alpha.csv')
Dataframe.SplitXY(Dataframe.df)
Dataframe.standardizeData()

# randomizeSplit(Dataframe)
kFoldCrossValidation(Dataframe)
