
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
        self.steps = 0
        print("READING STARTED")
        
        self.calc_steps_per_epoch()
        self.top = 0
        self.__read__()
        print("READ INIT")
        # self.on_epoch_end()

    def calc_steps_per_epoch(self):
        for mid in self.seq_names:
            filec = self.__next_file__()
            try : 
                ns = dataset.to_note_sequence(filec)
            except: continue
            ar = dataset.ns_to_tarray(ns, resolution=64)
            self.sFlat = dataset.ns_tarray_to_sFlat(t_arr= ar[ self.trk: self.trk + 1 ], DEPTH= self.DEPTH)
            self.steps_per_epoch += ((self.sFlat.shape[1] // self.batch_size) * self.batch_size + 1) // self.batch_size  - 1
                    

    def __next_file__(self):
        self.top += 1
        if self.top == len(self.seq_names) : return False
        # print("Top : ", self.top)
        return self.seq_names[self.top - 1]
    
    def __read__(self):
        # if not __exit__() : return not self.__exit__() 
        # filec = self.__next_file__()
        while not self.__exit__():
            try  :
                filec = self.__next_file__()
                ns = dataset.to_note_sequence(filec)
                break
            except: continue
                
        ar = dataset.ns_to_tarray(ns, resolution=64)
        self.sFlat = dataset.ns_tarray_to_sFlat(t_arr= ar[ self.trk: self.trk + 1 ], DEPTH= self.DEPTH)
        self.sFlat = self.sFlat[:, : (self.sFlat.shape[1] // self.batch_size) * self.batch_size + 1] 
        self.steps = self.sFlat.shape[1] // self.batch_size  - 1
        self.iter = 0

        self.flat_shape = self.sFlat.shape
        return True
    def __len__(self):
        return len(self.seq_names)

    def __exit__(self):
        if self.top == len(self.seq_names) : return True
        return False
    def on_epoch_end(self):
        self.top = 0

    def __getitem__(self, idx):
        # if not numpy.isscalar(self.sFlat): self.__read__()
        # if self.sFlat.shape[1] - self.iter - self.ip_memory - 1 < self.batch_size: 
        #     if not self.__read__(): return None, None
        if self.steps <= 0: self.__read__()
        
        enc = enc_deco.sFlat_to_octave(self.sFlat[:, self.iter : self.iter + self.batch_size + self.ip_memory])  #Improving started 
        x, y = dataset.prepare_sFlat_data(enc, enc_shape= enc.shape[-2: ], ip_memory=self.ip_memory, depth= self.DEPTH)
        # print(x.shape, y.shape, '----> x, y', self.flat_shape)
        x, y = numpy.reshape(x, x.shape[1:3] + (-1, )), numpy.reshape(y, y.shape[1:2] + (-1, )) #reshaping to fit as rnn input
        self.iter += self.batch_size
        self.steps -= 1
        # print("steps : ", self.steps)
        # print(x.shape, y.shape, '----> x, y')
        return x, y

    def __str__(self):
        return '{\n\ttrk : ' + str(self.trk) + "\n\tseq_name : " + str(self.seq_names) + "\n\tbatch_size : " + str(self.batch_size) + \
                "\n\tshape : " + str(self.shape) + '\n\tsFlat_shape : ' + str(self.flat_shape) + '\n\tsteps_per_epochs : ' + str(self.steps_per_epoch) + \
                '\n\titer : ' + str(self.iter) +'\n\tEND\n}'
        