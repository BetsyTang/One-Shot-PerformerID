import os
import torch
import itertools, os
import numpy as np
from progress.bar import Bar
from tqdm import tqdm

import config
import utils
from sequence import EventSeq, ControlSeq
from torch.utils.data import DataLoader


# pylint: disable=E1101
# pylint: disable=W0101

class Dataset:
    def __init__(self, root, verbose=True):
        assert os.path.isdir(root), root
        paths = utils.find_files_by_extensions(root, ['.data'])
        self.root = root
        self.samples = []
        self.seqlens = []
        self.performer_id = []
        self.title_id = []

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

        print(np.asarray(self.seqlens).shape)
    
    def pair(self, perform_id_seq, title_id_seq, train_list, test_list):
        triplets_train = []
        triplets_test = []
        for i in tqdm(range(len(perform_id_seq))):
            positive = np.where((perform_id_seq == perform_id_seq[i]) \
                & (title_id_seq != title_id_seq[i]))[0]
            positive = positive[positive>i]
            if perform_id_seq[i] in train_list:
                negative = np.where((perform_id_seq != perform_id_seq[i]) \
                    & (title_id_seq == title_id_seq[i]) \
                    & (perform_id_seq in train_list))[0]
                negative_expand = np.where((perform_id_seq != perform_id_seq[i]) \
                    & (perform_id_seq in train_list))[0]
                if negative.size == 0:
                    negative = negative_expand
                    # continue
                for j in range(len(positive)):
                    positive_choice = positive[j]
                    negative_choice = np.random.choice(negative)
                    triplets_train.append((i, positive_choice, negative_choice))
            else:
                negative = np.where((perform_id_seq != perform_id_seq[i]) \
                    & (title_id_seq == title_id_seq[i]) \
                    & (perform_id_seq in test_list))[0]
                negative_expand = np.where((perform_id_seq != perform_id_seq[i]) \
                    & (perform_id_seq in test_list))[0]
                if negative.size == 0:
                    negative = negative_expand
                    # continue
                for j in range(len(positive)):
                    positive_choice = positive[j]
                    negative_choice = np.random.choice(negative)
                    triplets_test.append((i, positive_choice, negative_choice))
        
        print(len(triplets_train), len(triplets_test))
        return triplets_train, triplets_test
    
    def sequence(self, window_size, stride_size):
        event_sequences = []
        control_sequences = []
        performer_id_sequences = []
        title_id_sequences = []
        for i, seqlen in tqdm(enumerate(np.asarray(self.seqlens))):
            eventseq, controlseq = self.samples[i]
            eventseq_batch = []
            controlseq_batch = []
            for j in range(0, seqlen-window_size, stride_size):
                event = eventseq[j:j+window_size]
                control = controlseq[j:j+window_size]
                eventseq_batch.append(event)
                controlseq_batch.append(control)
                if len(eventseq_batch) == 40:
                    event_sequences.append(np.stack(eventseq_batch, axis=0))
                    control_sequences.append(np.stack(controlseq_batch, axis=0))
                    performer_id_sequences.append(self.performer_id[i])
                    title_id_sequences.append(self.title_id[i])
                    
                    eventseq_batch = []
                    controlseq_batch = []
            return np.asarray(event_sequences), \
                    np.asarray(control_sequences), \
                    np.asarray(performer_id_sequences), \
                    np.asarray(title_id_sequences)
    
    def split(self):
        set_performer = np.asarray(list(set(self.performer_id)))
        np.random.seed(10086)
        train_performer = np.random.choice(set_performer, size=int(len(set_performer)*0.9), replace=False)
        test_performer = np.setdiff1d(set_performer, train_performer)
        return train_performer, test_performer

    
    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'samples={len(self.samples)}, '
                f'avglen={self.avglen})')


def generate_triplet_data_loader():
    data = Dataset("data_maestro/tmp")
    train_list, test_list = data.split()
    print("Start Generating Sequences...")
    event_list, control_list, performer_list, title_list = data.sequence(config.train['window_size'], \
                                config.train['stride_size'])
    print("Sequence Generating Done")
    print("Start Pairing...")
    triplets_train, triplets_test = data.pair(performer_list, title_list, train_list, test_list)
    print("Pairing Done")
    print("Start Making Triplets...")
    train_data = []; test_data = []
    for i in tqdm(range(len(triplets_train))):
        train_data.append(event_list[triplets_train[i],])
    for i in tqdm(range(len(triplets_test))):
        test_data.append(event_list[triplets_test[i],])
    np.save("train_data.npy", np.asarray(train_data))
    np.save("test_data.npy", np.asarray(test_data))
    print("Making Triplets Done")

    train_data = np.load("train_data.npy")
    train_data = train_data[np.random.choice(range(len(train_data)), size=50000, replace=False)]
    train_data = torch.LongTensor(train_data)
    train_data = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
    return train_data

if __name__ == '__main__':
    generate_triplet_data_loader()