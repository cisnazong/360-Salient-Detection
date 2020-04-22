import numpy as np
import self as self
import torch
import pandas as pd

# Function  create_dataset is abandoned, will be removed in future
def create_dataset(dataset:np.array, look_back=2, look_forward=1, divide_ratio=0.8):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_forward):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        b = dataset[(i+look_back):(i+look_back+look_forward)]
        dataY.append(b)

    train_size = int (len (dataX) * divide_ratio)
    test_size = len (dataX) - train_size

    return np.array(dataX[:train_size]), \
           np.array(dataY[:train_size]), \
           np.array(dataX[train_size:]), \
           np.array(dataY[train_size:])

class HeadPosDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 look_back=60,
                 look_forward=30,
                 preprocess=True,
                 column_names=['T','F','x','y','z','k'],
                 index=['x','y','z','k']):
        self.path = data_path
        data_df = pd.read_csv (data_path, ' ', names=column_names)
        if preprocess:
            data_df = data_df.dropna ()
            data_df = data_df.astype ('float32')

        data_np = data_df[index].values
        data_X, data_Y = [], []
        for i in range (len (data_np) - look_back - look_forward):
            a = data_np[i:(i + look_back)]
            data_X.append (a)
            b = data_np[(i + look_back):(i + look_back + look_forward)]
            data_Y.append (b)

        self.X = torch.from_numpy(np.array(data_X).reshape(-1, look_back,4))
        self.Y = torch.from_numpy(np.array(data_Y).reshape(-1, look_forward,4))

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return len(self.X)


