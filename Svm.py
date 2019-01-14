from sklearn.svm import SVC
import Database 
import numpy as np
from sklearn.model_selection import LeaveOneOut	#Cross Validation Leave one out method
from sklearn.metrics import classification_report,confusion_matrix  #Result and outcome

loo = LeaveOneOut()

def topDownApproach(Head_Data):

	loo.get_n_splits(Head_Data)

	svclassifier =SVC(kernel = 'linear')
	
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
	print('Report',classification_report(Dataframe.Result,y_pred))


Dataframe = Database.Dataset()

min_df = Dataframe.DoFeatureScaling()
min_df = Dataframe.DoLabelEncoding(min_df)
Dataframe.SplitXY(min_df) 
Dataframe.Data = Dataframe.EmputationMaximization()
# Dataframe.Data = Dataframe.standardizeData()

topDownApproach(Dataframe.Data)


