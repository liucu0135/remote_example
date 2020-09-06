import torch
import torch.nn as nn
import numpy as np
import argparse
import time

'''A small example for remote usage of the super computer'''


# Handle the arguments
parser = argparse.ArgumentParser(description='example')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='use gpu(default false)')

# A simple neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        layers=[
            nn.Linear(1,1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
        m.bias.data.fill_(0)

if __name__ == '__main__':
    # obtain args
    args = parser.parse_args()

    # Generate data for training and testing
    x = np.arange(1000)
    y = x*2 + 1

    x=torch.from_numpy(x.astype(np.float32))
    y=torch.from_numpy(y.astype(np.float32))

    x_train = x[:800]
    y_train = y[:800]

    x_test = x[800:]
    y_test = y[800:]

    net = Network()
    net.apply(weights_init_normal)
    opt=torch.optim.Adam(net.parameters(), lr=0.001)
    if args.gpu:
        print('using gpu...')
        x_train=x_train.cuda()
        y_train=y_train.cuda()
        x_test=x_test.cuda()
        y_test=y_test.cuda()
        net=net.cuda()
    else:
        print('using cpu...')


    # start training
    start=time.time()
    for epoch in range(500):
        dx=x_train.unsqueeze(1)
        py=net(dx)
        dy=y_train.unsqueeze(1)
        loss=torch.nn.functional.mse_loss(py,dy)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch%100==0:
            print('epoch {} finished, loss:{}'.format(epoch, loss))

    print('training finished with {}, time consumed:{}'.format({0:'cpu',1:'gpu'}[args.gpu], time.time()-start))





