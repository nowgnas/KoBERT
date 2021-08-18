from imports.packages import *
from model.KoBERT import *

path = '../data/'
train_data = pd.read_csv(path + 'train_data.csv', encoding='utf-8-sig')

train_dataset = []
for sen, label in zip(train_data['title'], train_data['topic_idx']):
    data_train = []
    data_train.append(sen)
    data_train.append(str(label))

    train_dataset.append(data_train)

