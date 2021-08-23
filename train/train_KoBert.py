import collections

import tqdm

from imports.packages import *
from model.KoBERT import *

# Setting parameters
max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 4
max_grad_norm = 1
log_interval = 200
learning_rate = 4e-5

path = '../data/'
train_data = pd.read_csv(path + 'train_data.csv', encoding='utf-8-sig')

train_dataset = []
for sen, label in zip(train_data['title'], train_data['topic_idx']):
    data_train = []
    data_train.append(sen)
    data_train.append(str(label))

    train_dataset.append(data_train)

# BERT 모델, Vocabulary 불러오기
bertModel, vocab = get_pytorch_kobert_model()

# 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

'''
data_train = BERTDataset(train_dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair)
'''
'''
train dataset
[['인천→핀란드 항공기 결항…휴가철 여행객 분통', '4'], ['실리콘밸리 넘어서겠다…구글 15조원 들여 美전역 거점화', '4']]
'''
data_train = BERTDataset(train_dataset, 0, 1, tok, max_len, True, False)

'''
data train: 
(array([   2, 3790, 2869, 5859, 4955, 6970, 7928, 1043, 7178, 6553, 2485,
        517,   55,  541, 6416, 6150, 6896, 4378,    3,    1,    1,    1,
          1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
          1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
          1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
          1,    1,    1,    1,    1,    1,    1,    1,    1], dtype=int32), array(19, dtype=int32), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=int32), 2)
'''
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)

# BERT 모델 불러오기
'''
show layer
'''
model = BERTClassifier(bertModel, dr_rate=0.4).to(device)

# optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
'''
any(): 
'''
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

'''-----------------------------------------train----------------------------------------------'''
for e in range(num_epochs):
    train_acc = 0.0
    # test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
        # optimizer initialize
        optimizer.zero_grad()

        # train loader
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        # print(f'token {token_ids} \n segment :{segment_ids}  \n valid lengh {label}')
        '''
        token tensor([[   2, 4652, 6122,  ...,    1,    1,    1],
        [   2,  517, 5256,  ...,    1,    1,    1],
        [   2, 4958, 1423,  ...,    1,    1,    1],
        ...,
        [   2, 4519, 7330,  ...,    1,    1,    1],
        [   2, 3574, 6710,  ...,    1,    1,    1],
        [   2, 2417, 7612,  ...,    1,    1,    1]], device='cuda:0') 
 segment :tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]], device='cuda:0')  
 valid lengh tensor([5, 6, 5, 5, 5, 4, 4, 5, 5, 6, 6, 5, 4, 4, 5, 5, 4, 4, 4, 4, 6, 4, 5, 6,
        5, 6, 4, 6, 4, 5, 6, 4], device='cuda:0')
        '''

        # model out
        out = model(token_ids, valid_length, segment_ids)
        # EX: goto KoBERT.py forward()

        # loss function
        loss = loss_fn(out, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)

        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                     train_acc / (batch_id + 1)))
    print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
