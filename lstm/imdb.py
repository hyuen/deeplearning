import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import bcolz


class RNNModel(nn.Module):
    def __init__(self, ntoken=20000, ninp=128, nhid=128, nlayers=1):
        super(RNNModel, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=0.2)
        self.dense = nn.Linear(nhid, 1)
        self.sm = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.dense.bias.data.fill_(0)
        self.dense.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        return (Variable(torch.zeros(1, 1, self.nhid).cuda()),
                Variable(torch.zeros(1, 1, self.nhid).cuda()))

    def forward(self, sentence, hidden):
        #print("sentence", sentence.size())
        embeds = self.encoder(sentence)
        #print("embeds", embeds.size())
        lstm_out, hidden = self.rnn(embeds, hidden)
        #print('lstm out', lstm_out.size(),[h.size() for h in hidden])

        lstm_out = lstm_out.select(1, len(sentence)-1).contiguous()
        lstm_out = lstm_out.view(-1, self.nhid)
        #print('lstm out', lstm_out.size(),[h.size() for h in hidden])

        decoded = self.dense(lstm_out) 
        #print('decoded', decoded.size())
        output = self.sm(decoded)
        #print("sigmoided", output.size())
        #print (output.size())
        return output, hidden


def get_batch(x, y, i, batch_size, evaluation=False):
    d = torch.from_numpy(x[i:i+batch_size]).float()
    t = torch.from_numpy(y[i:i+batch_size]).float().view(-1)
    data = Variable(d.cuda(), volatile=evaluation)
    target = Variable(d.cuda())

    return data, target

    
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
        print('target', output.size(), targets.view(-1,1).size())
        #print(output, targets)
        loss = criterion(output, targets.view(-1,1))
        loss.backward(retain_graph=True)
        print("loss", i, loss.data, total_loss)
        #continue
    
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-0.01, p.grad.data)

        total_loss += loss.data
        continue


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
    model.cuda()
    criterion = nn.BCELoss()
    train(model, criterion, x_train, y_train, x_test, y_test)


        
main()    
#r = RNNModel()                            
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
