import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import impyute.imputation.cs as imp 


class Dataset:
	df = pd.DataFrame()
	Data = pd.DataFrame()
	Result = pd.DataFrame()
	def __init__(self):
		self.df = pd.read_csv("F:\\Coding\\Machine Learning\\Research-Paper\\Database\\WPBC.csv",na_values = "?")

	def DoFeatureScaling(self): 	
		column_names = ['Outcome', 'MeanRadius', 'MeanPerimeter','MeanArea',  'SmoothnessSE', 'ConcavitySE', 'ConcavePointsSE','TumorSize', 'LymphNodeStatus']
		min_df = self.df.loc[:,column_names]	#Feature Scaling Columns Seperated
		return min_df

	def DoLabelEncoding(self,min_df): 		#Converting y class into integer values
		le = preprocessing.LabelEncoder()
		min_df['Outcome'] = le.fit_transform(min_df['Outcome'])	#Label Encoding 
		return min_df

	def SplitXY(self,min_df): 		#Splitting into two classes
		self.Result = min_df['Outcome']
		self.Data = min_df.loc[:,"MeanRadius":"LymphNodeStatus"]

	# def fillNaMean(self):
	# 	self.Data =  self.Data.fillna(self.Data.mean())

	# def fillNaMode(self):
	# 	Test = self.Data.copy(deep = True)
	# 	for column in Test:
	# 		Test[column].fillna(Test[column].mode()[0],inplace = True)
	# 	return Test

	def fillNaMedian(self): 	#use to fill na values
		self.Data =  self.Data.fillna(self.Data.median())

	def standardizeData(self):
		columns = [ 'MeanRadius', 'MeanPerimeter','MeanArea',  'SmoothnessSE', 'ConcavitySE', 'ConcavePointsSE','TumorSize', 'LymphNodeStatus']
		scaler = StandardScaler()
		scaler.fit(self.Data)
		self.Data = pd.DataFrame(data = scaler.transform(self.Data),columns = columns)

	def EmputationMaximization(self): 	#Use to fill na values
		columns = ['MeanRadius', 'MeanPerimeter','MeanArea',  'SmoothnessSE', 'ConcavitySE', 'ConcavePointsSE','TumorSize', 'LymphNodeStatus']
		return pd.DataFrame(data = np.array(imp.em(self.Data))  ,columns = columns)


# Dataframe = Dataset()
# Dataframe.df = Dataframe.df.dropna(subset = ['LymphNodeStatus'])
# Dataframe.df = Dataframe.DoFeatureScaling() 
# Dataframe.df = Dataframe.DoLabelEncoding(Dataframe.df)
# Dataframe.df.to_csv("alpha.csv", index = False)