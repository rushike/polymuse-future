
from polymuse import dataset, dataset2 as d2, constant, enc_deco

from keras.utils import Sequence

import numpy


class NoteDataGenerator(Sequence):
    def __init__(self, trk, seq_names, batch_size, ip_memory):
        self.seq_names = numpy.array(seq_names) # list of midi files avalable
        self.batch_size = batch_size # batch size used while training , i.e. no. of instances at time
        self.sFlat = None # strores the sFlat representation of midi file
        self.top = 0 #increases by every file
        self.trk = trk #which track (lead : 0, chorus : 1, drum : 2)
        self.DEPTH = constant.depths_of_3tracks[trk] # depth parrameter while making of sFlat...
        self.iter = 0 #Increases by every batch size
        self.ip_memory = ip_memory
        self.flat_shape = None
        self.shape = (batch_size, ip_memory, self.DEPTH * 32)
        self.steps_per_epoch = 0
        self.on_epoch_end()

    def __next_file__(self):
        self.top += 1
        return self.seq_names[self.top - 1]
    
    def __read__(self):        
        ns = dataset.to_note_sequence(self.__next_file__())
        ar = dataset.ns_to_tarray(ns, resolution=64)
        self.sFlat = dataset.ns_tarray_to_sFlat(t_arr= ar[ self.trk: self.trk + 1 ], DEPTH= self.DEPTH)
        self.sFlat = self.sFlat[:, : (self.sFlat.shape[1] // self.batch_size) * self.batch_size]
        self.steps_per_epoch = self.sFlat.shape[1] // self.batch_size + 3
        self.flat_shape = self.sFlat.shape
    def __len__(self):
        return len(self.seq_names) // self.batch_size
    
    def __getitem__(self, idx):
        if not numpy.isscalar(self.sFlat): self.__read__()
        if self.sFlat.shape[1] - self.iter - self.ip_memory - 2 < self.batch_size: 
            if not self.__read__(): return None
        
        enc = enc_deco.sFlat_to_octave(self.sFlat[idx * self.batch_size : (idx + 1) * self.batch_size + self.ip_memory])
        x, y = dataset.prepare_sFlat_data(enc, enc_shape= enc.shape[-2: ], ip_memory=self.ip_memory, depth= self.DEPTH)
        x, y = numpy.reshape(x, x.shape[:2] + (-1, )), numpy.reshape(x, x.shape[:2] + (-1, )) #reshaping to fit as rnn input
        print(x.shape, y.shape, '----> x, y')
        return x, y

    def __str__(self):
        return '{\n\ttrk : ' + str(self.trk) + "\n\tseq_name : " + str(self.seq_names) + "\n\tbatch_size : " + str(self.batch_size) + \
                "\n\tshape : " + str(self.shape) + '\n\tsFlat_shape : ' + str(self.flat_shape) + '\n\tsteps_per_epochs : ' + str(self.steps_per_epoch) +'\n\tEND\n}'
        