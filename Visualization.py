import Database
import matplotlib.pyplot as plt 

columns = ['Time','LymphNodeStatus','PermiterWorst','AreaSE','AreaWorst','MeanSmoothness','PerimeterSE','MeanPerimeter','TumorSize']


Dataframe = Database.Dataset()

min_df = Dataframe.DoFeatureScaling()
min_df = Dataframe.DoLabelEncoding(min_df)


Dataframe.SplitXY(min_df) 

Dataframe.Data = Dataframe.EmputationMaximization()

x = 3
y = 3

i = 0
for column in Dataframe.Data:
	ax = plt.subplot(x,y,i+1)
	plt.plot(Dataframe.Data[column])
	plt.legend([columns[i]])
	i+=1
plt.show()

stdData = Dataframe.standardizeData()

i = 0
for column in stdData:
	ax = plt.subplot(x,y,i+1)
	plt.plot(stdData[column])
	plt.legend([columns[i]])
	i+=1
plt.show()





