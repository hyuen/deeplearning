import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import bcolz

use_cuda = 1

class RNNModel(nn.Module):
    def __init__(self, ntoken=20000, ninp=128, nhid=128, nlayers=7):
        super(RNNModel, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(input_size=ninp, hidden_size=nhid, num_layers=nlayers, dropout=0.2)
        #self.gru = nn.GRU(input_size=nhid, hidden_size=64, dropout=0.2)
        #self.gru = nn.LSTM(nhid, 64, dropout=0.2)
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
        embeds = self.encoder(sentence)
        lstm_out, hidden = self.rnn(embeds, hidden)
        lstm_out = lstm_out[-1]
        decoded = self.dense(lstm_out)
        output = self.sm(decoded).view(-1)

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


def train(model, criterion, x_train, y_train, x_test, y_test, epoch, lr):
    model.train()
    total_loss = 0
    start_time = time.time()

    batch_size = 128
    reporting_size = batch_size

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for batch, i in enumerate(range(0, int(len(x_train)/batch_size * batch_size) -batch_size, batch_size)):
        data, targets = get_batch(x_train, y_train, i, batch_size)
        hidden = model.init_hidden(batch_size)
        optimizer.zero_grad()

        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.data

        if batch % reporting_size == 0 and batch > 0:
            cur_loss = total_loss[0] / reporting_size
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.4f} | ppl {:8.2f}'.format(
                        epoch, batch, 25000// 32, lr,
                        elapsed * 1000 / reporting_size,cur_loss, math.exp(cur_loss)))
            start_time = time.time()

            total_loss = 0

start_time = time.time()
def main():
    max_features = 20000

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
    lr = 0.1
    for epoch in range(15):
        train(model, criterion, x_train, y_train, x_test, y_test, epoch, lr)

if __name__ == "__main__":
    main()
#r = RNNModel()                            
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
