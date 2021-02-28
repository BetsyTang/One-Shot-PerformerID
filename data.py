import os
import torch
import itertools, os
import numpy as np
from progress.bar import Bar

import config
import utils
from sequence import EventSeq, ControlSeq
from torch.utils.data import DataLoader


# pylint: disable=E1101
# pylint: disable=W0101

class Dataset:
    def __init__(self, root, verbose=False):
        assert os.path.isdir(root), root
        paths = utils.find_files_by_extensions(root, ['.data'])
        self.root = root
        self.samples = []
        self.seqlens = []
        self.performer_id = []
        self.title_id = []
        self.triplets = []
        if verbose:
            paths = Bar(root).iter(list(paths))
        for path in paths:
            eventseq, controlseq = torch.load(path)
            controlseq = ControlSeq.recover_compressed_array(controlseq)
            assert len(eventseq) == len(controlseq)

            performer_id = os.path.basename(path).split("_")[0]
            title_id = os.path.basename(path).split("_")[1]

            self.performer_id.append(performer_id)
            self.title_id.append(title_id)
            self.samples.append((eventseq, controlseq))
            self.seqlens.append(len(eventseq))
        self.avglen = np.mean(self.seqlens)

        print(np.asarray(self.samples).shape)
        print(np.asarray(self.seqlens).shape)
    
    def pair(self):
        triplets = []
        for i in range(len(self.samples)):
            positive = np.where(np.asarray(self.performer_id) == self.performer_id[i])[0]
            positive = positive[positive>i]
            negative = np.where((np.asarray(self.performer_id) != self.performer_id[i]) \
            & (np.asarray(self.title_id) == self.title_id[i]))[0]
            negative_expand = np.where(np.asarray(self.performer_id) != self.performer_id[i])[0]
            if negative.size == 0:
                negative = negative_expand
            for j in range(len(positive)):
                positive_choice = positive[j]
                negative_choice = np.random.choice(negative)
                triplets.append((i, positive_choice, negative_choice))
        
        print(len(triplets))
        self.triplets = triplets
        print(self.performer_id[triplets[0][0]],self.performer_id[triplets[0][1]],self.performer_id[triplets[0][2]])
        return triplets
    
    def sequence(self, window_size, stride_size):
        event_sequences = []
        control_sequences = []
        for i, seqlen in enumerate(np.asarray(self.seqlens)):
            eventseq, controlseq = self.samples[i]
            eventseq_batch = []
            controlseq_batch = []
            for j in range(0, seqlen-window_size, stride_size):
                event = eventseq[j:j+window_size]
                control = controlseq[j:j+window_size]
                eventseq_batch.append(event)
                controlseq_batch.append(control)
            event_sequences.append(np.stack(eventseq_batch, axis=1))
            control_sequences.append(np.stack(controlseq_batch, axis=1))
        return event_sequences, control_sequences
    
    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'samples={len(self.samples)}, '
                f'avglen={self.avglen})')


def generate_triplet_data_loader():
    # data = Dataset("data_maestro/tmp")
    # triplets = data.pair()
    # print("Pairing Done")
    # event_list, control_list = data.sequence(config.train['window_size'], config.train['stride_size'])
    # triplet_data = [(torch.FloatTensor(np.swapaxes(event_list[triplets[i][0]], 0, 1)),
    # torch.FloatTensor(np.swapaxes(event_list[triplets[i][1]], 0, 1)),
    # torch.FloatTensor(np.swapaxes(event_list[triplets[i][2]], 0, 1))) for i in range(len(triplets))]
    # torch.save(triplet_data, 'triplet_data_expand.data')
    triplet_data = torch.load("triplet_data_expand.data")
    triplet_data = DataLoader(triplet_data, batch_size=1, shuffle=True)
    return triplet_data