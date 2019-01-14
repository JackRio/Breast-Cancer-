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
		self.df = pd.read_csv("F:\\Machine Learning\\Research-Paper\\Database\\WPBC.csv",na_values = "?")

	def DoFeatureScaling(self):
		column_names = ['Outcome','Time','LymphNodeStatus','PermiterWorst','AreaSE','AreaWorst','MeanSmoothness','PerimeterSE','MeanPerimeter','TumorSize']
		min_df = self.df.loc[:,column_names]	#Feature Scaling Columns Seperated
		return min_df

	def DoLabelEncoding(self,min_df):
		le = preprocessing.LabelEncoder()
		min_df['Outcome'] = le.fit_transform(min_df['Outcome'])	#Label Encoding 
		return min_df

	def SplitXY(self,min_df):
		self.Result = min_df['Outcome']
		self.Data = min_df.loc[:,"Time":"TumorSize"]

	def fillNaMean(self):
		self.Data =  self.Data.fillna(self.Data.mean())

	def fillNaMode(self):
		Test = self.Data.copy(deep = True)
		for column in Test:
			Test[column].fillna(Test[column].mode()[0],inplace = True)
		return Test

	def fillNaMedian(self):
		self.Data =  self.Data.fillna(self.Data.median())

	def standardizeData(self):
		columns = ['Time','LymphNodeStatus','PermiterWorst','AreaSE','AreaWorst','MeanSmoothness','PerimeterSE','MeanPerimeter','TumorSize']
		scaler = StandardScaler()
		scaler.fit(self.Data)
		return pd.DataFrame(data = scaler.transform(self.Data),columns = columns)

	def EmputationMaximization(self):
		columns = ['Time','LymphNodeStatus','PermiterWorst','AreaSE','AreaWorst','MeanSmoothness','PerimeterSE','MeanPerimeter','TumorSize']
		return pd.DataFrame(data = np.array(imp.em(self.Data))  ,columns = columns)

Dataframe = Dataset()



