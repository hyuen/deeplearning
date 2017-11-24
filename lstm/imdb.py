import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import bcolz

use_cuda = 1

class RNNModel(nn.Module):
    def __init__(self, ntoken=20000, ninp=128, nhid=128, nlayers=5):
        super(RNNModel, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(input_size=ninp, hidden_size=nhid, num_layers=nlayers, dropout=0.2)
        #        self.gru = nn.GRU(input_size=nhid, hidden_size=64, dropout=0.2)
        self.dense = nn.Linear(nhid, 1)
        self.sm = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.dense.bias.data.fill_(0)
        self.dense.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        if use_cuda:
            return (Variable(torch.zeros(self.nlayers, bsz, self.nhid).cuda()),
                    Variable(torch.zeros(self.nlayers, bsz, self.nhid).cuda()))
        else:
            return (Variable(torch.zeros(self.nlayers, bsz, self.nhid)),
                    Variable(torch.zeros(self.nlayers, bsz, self.nhid)))

    def forward(self, sentence, hidden):
        #print("sentence", sentence.size())
        embeds = self.encoder(sentence)
        #print ("embeds", embeds.size())
        lstm_out, hidden = self.rnn(embeds, hidden)
        #print("out", lstm_out.size())
        lstm_out = lstm_out.select(0, len(sentence)-1).contiguous()
        #print("out", lstm_out.size())

        decoded = self.dense(lstm_out)
        #print("decoded", decoded.size())
        output = self.sm(decoded).view(-1)
        #print("sigmoid", output.size())

        return output, hidden


def get_batch(x, y, i, batch_size, evaluation=False):
    d = torch.from_numpy(x[i:i+batch_size].transpose()).long()
    t = torch.from_numpy(y[i:i+batch_size].transpose()).float().view(-1)
    if use_cuda:
        d = d.cuda()
        t = t.cuda()
    data = Variable(d, volatile=evaluation)
    target = Variable(t)

    return data, target

lr = 0.01
def train(model, criterion, x_train, y_train, x_test, y_test):
    model.train()
    total_loss = 0
    start_time = time.time()

    batch_size = 32

    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, len(x_train), batch_size)):
        data, targets = get_batch(x_train, y_train, i, batch_size)

        model.zero_grad()
        output, hidden = model(data, hidden)
        #print("output_sz", output.size(), targets.size()) 
        loss = criterion(output, targets)
        loss.backward(retain_graph=True)
        #if i % 100 == 0:
    
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-lr , p.grad.data)

        total_loss += loss.data

        if batch % 10 == 0 and batch > 0:
            cur_loss = total_loss[0] / 2
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                1, batch, 25000// 32, lr,
                        elapsed * 1000 / 2,cur_loss, math.exp(cur_loss)))
            total_loss = 0

start_time = time.time()
def main():
    max_features = 20000
    #maxlen = 80  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    print('Loading data')
    x_train = bcolz.open(rootdir="x_train")
    y_train = bcolz.open(rootdir="y_train")
    x_test = bcolz.open(rootdir="x_test")
    y_test = bcolz.open(rootdir="y_test")
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)


    model = RNNModel()
    if use_cuda:
        model = model.cuda()
    criterion = nn.BCELoss()
    train(model, criterion, x_train, y_train, x_test, y_test)


        
main()    
#r = RNNModel()                            
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
