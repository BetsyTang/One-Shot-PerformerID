import os
import re
import sys
import torch
import hashlib
import pandas as pd
from progress.bar import Bar
from concurrent.futures import ProcessPoolExecutor

from sequence import NoteSeq, EventSeq, ControlSeq
import utils
import config

def preprocess_midi(path):
    note_seq = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    control_seq = ControlSeq.from_event_seq(event_seq)
    return event_seq.to_array(), control_seq.to_compressed_array()
    
def preprocess_midi_files_under(csv_path, save_dir, num_workers):
    csv_file = pd.read_csv(csv_path, header=0)
    midi_paths = 'data_maestro/maestro-v3.0.0/'+ csv_file['midi_filename']
    midi_names = csv_file['performer_label'].apply(str)+"_"+csv_file['canonical_title_label'].apply(str)
    midi_paths = pd.concat([midi_paths, midi_names], axis=1).values.tolist()
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}_{}.data'
    results = []
    executor = ProcessPoolExecutor(num_workers)

    for path in midi_paths:
        try:
            results.append((path, executor.submit(preprocess_midi, path[0])))
        except KeyboardInterrupt:
            print(' Abort')
            return
        except:
            print(' Error')
            continue
    
    for path, future in Bar('Processing').iter(results):
        print(' ', end='[{}]'.format(path[0]), flush=True)
        name = path[1]
        code = hashlib.md5(path[0].encode()).hexdigest()
        save_path = os.path.join(save_dir, out_fmt.format(name, code))
        torch.save(future.result(), save_path)

    print('Done')

if __name__ == '__main__':
    preprocess_midi_files_under(
            csv_path=sys.argv[1],
            save_dir=sys.argv[2],
            num_workers=int(sys.argv[3]))
