from imports.packages import *

# GPU 사용
device = torch.device("cuda:0")


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        # https://nlp.gluon.ai/api/data.html#gluonnlp.data.BERTSentenceTransform
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,  ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        '''
        attention mask shape : [32, 64]
        '''
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        '''
        tensor([[1., 1., 1.,  ..., 0., 0., 0.],
        [1., 1., 1.,  ..., 0., 0., 0.],
        [1., 1., 1.,  ..., 0., 0., 0.],
        ...,
        [1., 1., 1.,  ..., 0., 0., 0.],
        [1., 1., 1.,  ..., 0., 0., 0.],
        [1., 1., 1.,  ..., 0., 0., 0.]], device='cuda:0')
        '''

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        '''
        pooler
        tensor([[-0.0347, -0.0237,  0.1961,  ...,  0.0589, -0.0124, -0.0040],
        [-0.0019, -0.0757,  0.3460,  ...,  0.1962, -0.0281, -0.0877],
        [-0.0697, -0.0337, -0.2580,  ...,  0.0254,  0.0333, -0.0956],
        ...,
        [ 0.0161, -0.0539,  0.1538,  ...,  0.0532, -0.0290, -0.0034],
        [ 0.0332, -0.1109,  0.1666,  ...,  0.0911, -0.0191, -0.0102],
        [-0.0591, -0.0192,  0.0081,  ...,  0.0039, -0.0573, -0.0765]],
       device='cuda:0', grad_fn=<TanhBackward>)
        '''
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


# 정확도 측정을 위한 함수 정의
def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc
