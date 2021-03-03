import torch
from network import *
from config import *
from torch import optim

import numpy as np
import os
import sys
import glob
import time
import optparse
import config
from data import *
from sequence import NoteSeq, EventSeq, ControlSeq

torch.backends.cudnn.benchmark = True

print("Loading data")
triplet_data = generate_triplet_data_loader(generate=True)
print(len(triplet_data))
print("Building network")
net = TripletNet(OSRNN(240,256,3)).to(device)
# net = TripletNet(EventSequenceEncoder(hidden_dim=256)).to(device)
# net = torch.load("final_model.pt")
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.TripletMarginLoss()

def checkpoint(net, save_path, loss, iterations):
    snapshot_prefix = os.path.join(save_path, 'checkpoint_' + net._class_name())
    snapshot_path = snapshot_prefix + '_loss_{:.4f}_iter_{}_model.pt'.format(loss, iterations)
    torch.save(net, snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)

iterations = 0


header = '  Time Epoch Iteration Progress    (%Epoch)   Loss'
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.1f}%,{:>7.4f}'.split(','))

print("Start training...")
print(header)
start = time.time()
for epoch in range(50):  # loop over the dataset multiple times
    print_loss = 0
    n = 0
    for i, data in enumerate(triplet_data, 0):
        iterations += 1
        # get the inputs
        inputs = data.to(device)
        inputs = torch.reshape(inputs, inputs.shape[1:])
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output_1,output_2,output_3 = net.forward(inputs[0],inputs[1],inputs[2])
        loss = criterion(output_1, output_2, output_3) 
        
        print_loss += loss.item()
        n += 1
        
        loss.backward()
        optimizer.step()  

        if iterations % len(triplet_data) == 0:
            checkpoint(net, "log/", print_loss/n, iterations)
            print(log_template.format(time.time()-start,
                    epoch, iterations, 1+i, len(triplet_data),
                    100. * (1+i) / len(triplet_data), print_loss/n))

torch.save(net, "final_model.pt")