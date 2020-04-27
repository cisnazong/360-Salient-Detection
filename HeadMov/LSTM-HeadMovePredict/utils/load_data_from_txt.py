import pandas as pd
import matplotlib.pyplot as plt
def load_data_from_txt(data_path:str,
                       preprocess=True,
                       normalize=False,
                       show=True,
                       column_names=['T','F','x','y','z','k'],
                       offset=2,
                       index=['x','y','z','k']):
	# use pandas to read
	data_df = pd.read_csv(data_path,' ',names=column_names)

	# prepare the data
	if preprocess:
		data_df = data_df.dropna()
		data_df = data_df.astype('float32')
	if normalize:
		pass

	#show the data
	if show:
		plt.title('origin')
		plt.ylim (-1.1, 1.1)
		plt.plot(data_df[column_names[0]],data_df[column_names[offset]],color='b')
		plt.plot(data_df[column_names[0]],data_df[column_names[offset+1]],color='g')
		plt.plot(data_df[column_names[0]],data_df[column_names[offset+2]],color='y')
		plt.plot(data_df[column_names[0]],data_df[column_names[offset+3]],color='r')
		plt.show()

	# create a data set from values
	assert isinstance (data_df,pd.DataFrame )
	return  data_df[index].values