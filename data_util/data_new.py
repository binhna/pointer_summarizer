import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/'.join(dir_path.split('/')[:-1])
sys.path.append(dir_path)

from torchtext.data import Field, Example, Dataset
from torchtext.vocab import Vectors
import torch
from torch.autograd import Variable
import json
from tqdm import tqdm
from collections import Counter
from torchtext.data import BucketIterator
import pickle
from training_ptr_gen.model import Model
import numpy as np
from sklearn.model_selection import train_test_split
import glob


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

# from data_util.config import batch_size, hidden_dim
from data_util import config

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


def prepare_data_cnn(path):
    files = glob.glob(os.path.join(path, '*.story'))
    data = []
    for file in tqdm(files):
        with open(file, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        src = []
        tgt = []
        target = False
        for i, line in enumerate(lines):
            if line == '@highlight':
                target = True
                continue
            if not target:
                src.append(line)
            else:
                tgt.append(line) 
        data.append({'src': ' '.join(src), 'tgt': ' '.join(tgt), 'id': os.path.basename(file).split('.')[0]})
    return data



def train():
    target_field = Field(sequential=True,
                        init_token=START_DECODING,
                        eos_token=STOP_DECODING,
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
    path = '../data/cnn_stories_tokenized'

    # train_src, train_tgt, train_id = load_data(train_path)
    # dev_src, dev_tgt, dev_id = load_data(dev_path)
    # test_src, test_tgt, test_id = load_data(test_path)
    train_data = prepare_data_cnn(path)
    # print(train_data[0])
    train_src = [dt['src'] for dt in train_data]
    train_tgt = [dt['tgt'] for dt in train_data]
    train_id = [dt['id'] for dt in train_data]
    train_src, test_src, train_tgt, test_tgt = train_test_split(train_src, train_tgt, test_size=0.15, random_state=123)
    train_id, test_id = train_test_split(train_id, test_size=0.15, random_state=123)
    # print(f"{len(train_src)}, {len(train_tgt)}")
    train_src, dev_src, train_tgt, dev_tgt = train_test_split(train_src, train_tgt, test_size=0.15, random_state=123)
    train_id, dev_id = train_test_split(train_id, test_size=0.15, random_state=123)

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
        batch_sizes=(config.batch_size, config.batch_size, config.batch_size),
        device=device,
        sort_key=lambda x: len(x.source),
        sort_within_batch=True)

    model = Model()

    for batch in tqdm(train_iter):
        batch_size = batch.batch_size
        #encoder part
        enc_padding_mask = get_mask(batch.source, device)
        enc_batch = batch.source[0]
        enc_lens = batch.source[1]
        encoder_outputs, encoder_feature, encoder_hidden = model.encoder(enc_batch, enc_lens)
        s_t_1 = model.reduce_state(encoder_hidden)
        coverage = Variable(torch.zeros(batch.source[0].size())).to(device)
        c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim))).to(device)
        extra_zeros, enc_batch_extend_vocab, max_art_oovs = get_extra_features(batch.source[0], source_field.vocab)
        extra_zeros = extra_zeros.to(device)
        enc_batch_extend_vocab = enc_batch_extend_vocab.to(device)
        #decoder part
        dec_batch = batch.target[0][:, :-1]
        # print(dec_batch.size())
        target_batch = batch.target[0][:, 0:]
        dec_lens_var = batch.target[1]
        dec_padding_mask = get_mask(batch.target, device)
        max_dec_len = max(dec_lens_var)
        
        

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps) - 1):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        # norm = clip_grad_norm_(model.encoder.parameters(), config.max_grad_norm)
        # clip_grad_norm_(model.decoder.parameters(), config.max_grad_norm)
        # clip_grad_norm_(model.reduce_state.parameters(), config.max_grad_norm)

        # optimizer.step()

        # return loss.item()




def get_mask(batch, device):
    # print('each batch', batch[0].size())
    maxlen = batch[0].size()[1]
    max_enc_seq_len = batch[1]
    mask = torch.arange(maxlen).to(device)
    mask = mask[None, :] < max_enc_seq_len[:, None]
    # print(batch.source[0]*mask)
    return mask

def get_extra_features(batch, vocab):
    unk_index = vocab.stoi[UNKNOWN_TOKEN]
    batch = batch.cpu().detach().numpy()
    batch_size = batch.shape[0]
    max_art_oovs = max([Counter(sample)[unk_index] for sample in batch])
    extra_zeros = None
    
    enc_batch_extend_vocab = np.full_like(batch, fill_value=vocab.stoi[PAD_TOKEN])
    max_art_oovs = 0
    for i, sample_index in enumerate(batch):
        oov_word_count = len(vocab)
        for j, word_index in enumerate(sample_index):
            if word_index == unk_index:
                enc_batch_extend_vocab[i, j] = oov_word_count
                oov_word_count += 1
        max_art_oovs = max(max_art_oovs, oov_word_count)
    max_art_oovs -= len(vocab)
    enc_batch_extend_vocab = Variable(torch.from_numpy(enc_batch_extend_vocab).long())

    extra_zeros = Variable(torch.zeros((batch_size, max_art_oovs)))
    return extra_zeros, enc_batch_extend_vocab, max_art_oovs


# def post_process(batch, vocab):
#     unk_index = vocab.stoi[UNKNOWN_TOKEN]
#     max_art_oovs = max([Counter(sample)[unk_token] for sample in batch])

#     for i, sample in enumerate(batch):
    
#         for j, index in enumerate(sample):
#             if index == vocab.stoi[]
train()

# print(next(iter(train_iter)))
# print(dev_data[10].source)
# if __name__ == "__main__":
#     for name in ['train', 'dev', 'test']:
#         process_incar_data(f'../data/{name}_public.json')
#     # vocabs = read_vocabs('../data/finished_files/vocab')
#     # print(len(vocabs))
