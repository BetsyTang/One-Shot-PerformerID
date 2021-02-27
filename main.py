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
triplet_data = generate_triplet_data_loader()

print("Building network")
net = TripletNet(OSRNN(200,512,3)).to(device)
optimizer = optim.Adam(net.parameters())
criterion = torch.nn.TripletMarginLoss()

def checkpoint(net, save_path, loss, iterations):
    snapshot_prefix = os.path.join(save_path, 'checkpoint_' + net._class_name())
    snapshot_path = snapshot_prefix + '_loss_{:.4f}_iter_{}_model.pt'.format(loss.item(), iterations)
    torch.save(net, snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)

iterations = 0

print("Start training...")
for epoch in range(50):  # loop over the dataset multiple times
    print_loss = 0
    n = 0
    for i, data in enumerate(triplet_data, 0):
        iterations += 1
        # get the inputs
        inputs = data
        # print(inputs.shape)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output_1,output_2,output_3 = net.forward(inputs[0].to(device),inputs[1].to(device),inputs[2].to(device))
        loss = criterion.forward(output_1, output_2, output_3) 
        print_loss += loss
        n += 1
        loss.backward()
        optimizer.step()  

        if iterations % 100 == 0:
            checkpoint(net, "log/", print_loss/n, iterations)

torch.save(net, "final_model.pt")