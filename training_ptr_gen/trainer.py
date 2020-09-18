import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/'.join(dir_path.split('/')[:-1])
sys.path.append(dir_path)

import torch
from torch.optim import Adagrad
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from torchtext.data import BucketIterator

from collections import Counter
import numpy as np
from tqdm import tqdm
import time

# from data_util.data_new import PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING
from data_util.utils import calc_running_avg_loss
from data_util.config import summary_path, model_dir

summary_writer = SummaryWriter(summary_path)

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

class Trainer():
    def __init__(
        self, 
        model, 
        args, 
        train_dataset, 
        eval_dataset, 
        test_dataset,
        vocab,
        is_train = True):
        self.model = model#.to(args.device)
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.is_train = is_train
        self.vocab = vocab

        self.params = list(model.encoder.parameters()) + \
            list(model.decoder.parameters()) + list(model.reduce_state.parameters())
        initial_lr = args.lr_coverage if args.is_coverage else args.lr
        self.optimizer = Adagrad(self.params, lr=initial_lr, 
                                initial_accumulator_value=args.adagrad_init_acc)
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')
        return BucketIterator(
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            device=self.args.device,
            sort_key=lambda x: len(x.source),
            sort_within_batch=True)
        
    def get_eval_dataloader(self):
        if self.eval_dataset is None:
            raise ValueError('Trainer: eval requires a eval_dataset.')
        return BucketIterator(
            dataset=self.eval_dataset,
            batch_size=self.args.batch_size,
            device=self.args.device,
            sort_key=lambda x: len(x.source),
            sort_within_batch=True)
    
    def get_test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError('Trainer: testing requires a test_dataset.')
        return BucketIterator(
            dataset=self.test_dataset,
            batch_size=self.args.batch_size,
            device=self.args.device,
            sort_key=lambda x: len(x.source),
            sort_within_batch=True)
    

    def get_mask(self, batch):
        # print('each batch', batch[0].size())
        maxlen = batch[0].size()[1]
        max_enc_seq_len = batch[1]
        mask = torch.arange(maxlen).to(self.args.device)
        mask = mask[None, :] < max_enc_seq_len[:, None]
        # print(batch.source[0]*mask)
        return mask


    def get_extra_features(self, batch):
        unk_index = self.vocab.stoi[UNKNOWN_TOKEN]
        batch = batch.cpu().detach().numpy()
        batch_size = batch.shape[0]
        max_art_oovs = max([Counter(sample)[unk_index] for sample in batch])
        extra_zeros = None

        enc_batch_extend_vocab = np.full_like(
            batch, fill_value=self.vocab.stoi[PAD_TOKEN])
        max_art_oovs = 0
        for i, sample_index in enumerate(batch):
            oov_word_count = len(self.vocab)
            for j, word_index in enumerate(sample_index):
                if word_index == unk_index:
                    enc_batch_extend_vocab[i, j] = oov_word_count
                    oov_word_count += 1
            max_art_oovs = max(max_art_oovs, oov_word_count)
        max_art_oovs -= len(self.vocab)
        enc_batch_extend_vocab = Variable(
            torch.from_numpy(enc_batch_extend_vocab).long())

        extra_zeros = Variable(torch.zeros((batch_size, max_art_oovs)))
        return extra_zeros, enc_batch_extend_vocab, max_art_oovs
    

    def save_model(self, running_avg_loss, iter, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss,
            'vocab': self.vocab
        }
        model_save_path = os.path.join(model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)


    def evaluate(self, eval_dataset=None, iter=0, is_test=False):
        if is_test:
            eval_iter = self.get_test_dataloader()
        else:
            eval_iter = self.get_eval_dataloader()
        self.model.eval()
        
        running_avg_loss = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_iter), total=len(eval_iter)):
                # print(batch.source[0].size())
                # exit()
                batch_size = batch.batch_size
                # encoder part
                enc_padding_mask = self.get_mask(batch.source)
                enc_batch = batch.source[0]
                enc_lens = batch.source[1]
                encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(
                    enc_batch, enc_lens)
                s_t_1 = self.model.reduce_state(encoder_hidden)
                coverage = Variable(torch.zeros(batch.source[0].size())).to(self.args.device)
                c_t_1 = Variable(torch.zeros(
                    (batch_size, 2 * self.args.hidden_dim))).to(self.args.device)
                extra_zeros, enc_batch_extend_vocab, max_art_oovs = self.get_extra_features(batch.source[0])
                extra_zeros = extra_zeros.to(self.args.device)
                enc_batch_extend_vocab = enc_batch_extend_vocab.to(self.args.device)
                # decoder part
                dec_batch = batch.target[0][:, :-1]
                # print(dec_batch.size())
                target_batch = batch.target[0][:, 0:]
                dec_lens_var = batch.target[1]
                dec_padding_mask = self.get_mask(batch.target)
                max_dec_len = max(dec_lens_var)

                step_losses = []
                for di in range(min(max_dec_len, self.args.max_dec_steps) - 1):
                    y_t_1 = dec_batch[:, di]  # Teacher forcing
                    final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                            encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                                                            extra_zeros, enc_batch_extend_vocab,
                                                                                            coverage, di)
                    target = target_batch[:, di]
                    gold_probs = torch.gather(
                        final_dist, 1, target.unsqueeze(1)).squeeze()
                    step_loss = -torch.log(gold_probs + self.args.eps)
                    if self.args.is_coverage:
                        step_coverage_loss = torch.sum(
                            torch.min(attn_dist, coverage), 1)
                        step_loss = step_loss + self.args.cov_loss_wt * step_coverage_loss
                        coverage = next_coverage

                    step_mask = dec_padding_mask[:, di]
                    step_loss = step_loss * step_mask
                    step_losses.append(step_loss)
                sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
                batch_avg_loss = sum_losses/dec_lens_var
                loss = torch.mean(batch_avg_loss)

                norm = clip_grad_norm_(
                    self.model.encoder.parameters(), self.args.max_grad_norm)
                clip_grad_norm_(self.model.decoder.parameters(), self.args.max_grad_norm)
                clip_grad_norm_(self.model.reduce_state.parameters(), self.args.max_grad_norm)

                self.optimizer.step()

                # running_avg_loss = loss if running_avg_loss == 0 else running_avg_loss * decay + (1 - decay) * loss
                # running_avg_loss = min(running_avg_loss, 12)
            name = 'Test' if is_test else 'Evaluation'
            calc_running_avg_loss(loss.item(), running_avg_loss, summary_writer, iter, name)
                # iter += 1


    # def predict(self, source_sentence):



    def train(self, model_path=None):


        train_iter = self.get_train_dataloader()
        iter, running_avg_loss = 0, 0
        start = time.time()
        for epoch in range(self.args.epoches):
            print(f"Epoch: {epoch+1}")
            self.model.train()
            for i, batch in tqdm(enumerate(train_iter), total=len(train_iter)):
                # print(batch.source[0].size())
                # exit()
                batch_size = batch.batch_size
                # encoder part
                enc_padding_mask = self.get_mask(batch.source)
                enc_batch = batch.source[0]
                enc_lens = batch.source[1]
                encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(
                    enc_batch, enc_lens)
                s_t_1 = self.model.reduce_state(encoder_hidden)
                coverage = Variable(torch.zeros(batch.source[0].size())).to(self.args.device)
                c_t_1 = Variable(torch.zeros(
                    (batch_size, 2 * self.args.hidden_dim))).to(self.args.device)
                extra_zeros, enc_batch_extend_vocab, max_art_oovs = self.get_extra_features(batch.source[0])
                extra_zeros = extra_zeros.to(self.args.device)
                enc_batch_extend_vocab = enc_batch_extend_vocab.to(self.args.device)
                # decoder part
                dec_batch = batch.target[0][:, :-1]
                # print(dec_batch.size())
                target_batch = batch.target[0][:, 0:]
                dec_lens_var = batch.target[1]
                dec_padding_mask = self.get_mask(batch.target)
                max_dec_len = max(dec_lens_var)

                step_losses = []
                for di in range(min(max_dec_len, self.args.max_dec_steps) - 1):
                    y_t_1 = dec_batch[:, di]  # Teacher forcing
                    final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                            encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                                                            extra_zeros, enc_batch_extend_vocab,
                                                                                            coverage, di)
                    target = target_batch[:, di]
                    gold_probs = torch.gather(
                        final_dist, 1, target.unsqueeze(1)).squeeze()
                    step_loss = -torch.log(gold_probs + self.args.eps)
                    if self.args.is_coverage:
                        step_coverage_loss = torch.sum(
                            torch.min(attn_dist, coverage), 1)
                        step_loss = step_loss + self.args.cov_loss_wt * step_coverage_loss
                        coverage = next_coverage

                    step_mask = dec_padding_mask[:, di]
                    step_loss = step_loss * step_mask
                    step_losses.append(step_loss)
                sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
                batch_avg_loss = sum_losses/dec_lens_var
                loss = torch.mean(batch_avg_loss)

                loss.backward()

                norm = clip_grad_norm_(
                    self.model.encoder.parameters(), self.args.max_grad_norm)
                clip_grad_norm_(self.model.decoder.parameters(), self.args.max_grad_norm)
                clip_grad_norm_(self.model.reduce_state.parameters(), self.args.max_grad_norm)

                self.optimizer.step()

                running_avg_loss = calc_running_avg_loss(loss.item(), running_avg_loss, summary_writer, iter, 'Train')
                iter += 1
                if iter % self.args.flush:
                    # print('flush')
                    summary_writer.flush()
                # print_interval = 10
                # if iter % print_interval == 0:
                #     print(f'steps {iter}, batch number: {i} with {time.time() - start} seconds, loss: {loss}')
                #     start = time.time()
                # if iter % 300 == 0:
            self.save_model(running_avg_loss, iter, model_dir)
            self.evaluate(self.eval_dataset, epoch)
            self.evaluate(self.test_dataset, epoch, True)