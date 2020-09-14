import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/'.join(dir_path.split('/')[:-1])
sys.path.append(dir_path)

from torchtext.data import Field, Example, Dataset
from torchtext.vocab import Vectors
import torch
import json
from tqdm import tqdm
from collections import Counter
from torchtext.data import BucketIterator
import pickle
from training_ptr_gen.model import Model


# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
PAD_TOKEN = '[PAD]'
# This has a vocab id, which is used to represent out-of-vocabulary words
UNKNOWN_TOKEN = '[UNK]'
# This has a vocab id, which is used at the start of every decoder input sequence
START_DECODING = '[START]'
# This has a vocab id, which is used at the end of untruncated target sequences
STOP_DECODING = '[STOP]'

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


def read_vocabs(vocab_file):
    vocabs = []
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        line_split = line.split()
        if len(line_split) == 2:
            vocabs.append(line_split[0])
    return vocabs


def process_incar_data(path, stat=False):
    utterance_data = []
    with open(path, 'r') as f:
        data = json.load(f)
    count_none = 0
    for sample in tqdm(data):
        num_sample = [1 for i in sample['dialogue'] if 'reformulation' in i and i['reformulation']
                      ['reformulated_utt'] and 'mturk_reformulations' in i['reformulation']]
        if not num_sample:
            count_none += 1
            # print(sample['scenario']['uuid'])
        for _ in num_sample:
            data_point = {'src': [], 'tgt': [],
                          'id': sample['scenario']['uuid']}
            previous_turn = False
            for utterance in sample['dialogue']:
                current_turn = utterance['turn']
                if current_turn != previous_turn:
                    pass
                else:
                    continue
                if 'reformulation' in utterance and utterance['reformulation']['reformulated_utt']:
                    if utterance_data and utterance['reformulation']['reformulated_utt'] in utterance_data[-1]['tgt']:
                        data_point['src'][-1] = utterance['reformulation']['reformulated_utt']
                        data_point['src'].append(
                            utterance['data']['utterance'])
                    else:
                        data_point['tgt'].append(
                            utterance['reformulation']['reformulated_utt'])
                        break
                else:
                    data_point['src'].append(utterance['data']['utterance'])
                previous_turn = current_turn
            utterance_data.append(data_point)
    print(f"{count_none} samples with no reformulation")
    # print(Counter([len(d['src']) for d in utterance_data]))
    path_out = os.path.splitext(path)[0]+'.pickle'
    with open(path_out, "wb") as f:
        pickle.dump(utterance_data, f)
    return utterance_data
    # utterance_data.append(sample['dialogue']['data']['utterance'])


class Mydataset(Dataset):
    def __init__(self, data, fields):
        super(Mydataset, self).__init__(
            [Example.fromlist([d['src'], d['tgt']], fields) for d in data],
            fields
        )


def load_data(path):
    with open(path, "rb") as f:
        utterance_data = pickle.load(f)
    # dataset = []
    src, tgt, id = [], [], []
    for data_point in tqdm(utterance_data):
        # dataset.append({'src': ' '.join(data_point['src']),
        #                 'tgt': data_point['tgt'],
        #                 'id': data_point['id']})
        src.append(' '.join(data_point['src']))
        tgt.append(data_point['tgt'][0])
        id.append(data_point['id'])
    return src, tgt, id


target_field = Field(sequential=True,
                     init_token=SENTENCE_START,
                     eos_token=SENTENCE_END,
                     pad_token=PAD_TOKEN,
                     batch_first=True,
                     include_lengths=True,
                     unk_token=UNKNOWN_TOKEN,
                     lower=True)

source_field = Field(sequential=True,
                     init_token=SENTENCE_START,
                     eos_token=SENTENCE_END,
                     pad_token=PAD_TOKEN,
                     batch_first=True,
                     include_lengths=True,
                     unk_token=UNKNOWN_TOKEN,
                     lower=True)
train_path = '../data/train_public.pickle'
dev_path = '../data/dev_public.pickle'
test_path = '../data/test_public.pickle'

train_src, train_tgt, train_id = load_data(train_path)
dev_src, dev_tgt, dev_id = load_data(dev_path)
test_src, test_tgt, test_id = load_data(test_path)


train_src_preprocessed = [source_field.preprocess(x) for x in train_src]
dev_src_preprocessed = [source_field.preprocess(x) for x in dev_src]
test_src_preprocessed = [source_field.preprocess(x) for x in test_src]

train_tgt_preprocessed = [target_field.preprocess(x) for x in train_tgt]
dev_tgt_preprocessed = [target_field.preprocess(x) for x in dev_tgt]
test_tgt_preprocessed = [target_field.preprocess(x) for x in test_tgt]
# train_src_preprocessed = source_field.apply(lambda x: source_field.preprocess(x))

vectors = Vectors(name='/home/binhna/Downloads/shared_resources/cc.en.300.vec',
                  cache='/home/binhna/Downloads/shared_resources/')

source_field.build_vocab(
    [train_src_preprocessed, dev_src_preprocessed,
     train_tgt_preprocessed, dev_tgt_preprocessed], vectors=vectors)
target_field.build_vocab(
    [train_src_preprocessed, dev_src_preprocessed,
     train_tgt_preprocessed, dev_tgt_preprocessed], vectors=vectors)

train_data = [{'src': src, 'tgt': tgt, 'id': id}
              for src, tgt, id in zip(train_src, train_tgt, train_id)]
train_data = Mydataset(data=train_data,
                       fields=(('source', source_field),
                               ('target', target_field)))
dev_data = [{'src': src, 'tgt': tgt, 'id': id}
            for src, tgt, id in zip(dev_src, dev_tgt, dev_id)]
dev_data = Mydataset(data=dev_data,
                     fields=(('source', source_field),
                             ('target', target_field)))

test_data = [{'src': src, 'tgt': tgt, 'id': id}
             for src, tgt, id in zip(test_src, test_tgt, test_id)]
test_data = Mydataset(data=test_data,
                      fields=(('source', source_field),
                              ('target', target_field)))
# print(train_data[10].source)
# print(train_data[10].target)
# print(len(target_field.vocab))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_iter, test_iter, dev_iter = BucketIterator.splits(
    datasets=(train_data, test_data, dev_data), 
    batch_sizes=(64, 64, 64),
    device=device,
    sort_key=lambda x: len(x.source),
    sort_within_batch=True)


model = Model()

for batch in tqdm(train_iter):
    model.encoder(batch.source[0], batch.source[1])
# print(next(iter(train_iter)))
# print(dev_data[10].source)
# if __name__ == "__main__":
#     for name in ['train', 'dev', 'test']:
#         process_incar_data(f'../data/{name}_public.json')
#     # vocabs = read_vocabs('../data/finished_files/vocab')
#     # print(len(vocabs))
