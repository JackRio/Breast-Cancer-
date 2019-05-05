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
	print('Report',classification_report(Dataframe.Result,y_pred)) # Check if you could use Guassian kernel


Dataframe = Database.Dataset() 						# Crating a object of class Datast which is imported above
min_df = Dataframe.DoFeatureScaling()				# Seperating and picking up columns which aare needed more info is in Database.py
min_df= min_df.dropna(subset = ['LymphNodeStatus']) # Dropping instances which have NA values 4 instances dropped here
min_df = Dataframe.DoLabelEncoding(min_df)			# Converting the values of outcome to 1/0 from N/R
Dataframe.SplitXY(min_df) 							# Splitting into Dataframe.Result("Outcome column"),Dataframe.Data("All the features") 
Dataframe.Data = Dataframe.standardizeData()		# Standardizing the data
topDownApproach(Dataframe.Data)						# Calling the above function to apply Leave One Out method on Dataset