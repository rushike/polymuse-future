
from polymuse import dataset, dataset2 as d2, constant, enc_deco

from keras.utils import Sequence

import numpy, random, traceback, sys

from sklearn.model_selection import train_test_split

"""
It generates the note data batch wise
Returns:
    NoteDataGenerator -- generator class for note while trainin
"""
class NoteDataGenerator(Sequence):
    def __init__(self, trk, seq_names, batch_size, ip_memory, enc = True, steps_per_epoch = 0, norm = True):
        self.seq_names = numpy.array(seq_names) # list of midi files avalable
        self.batch_size = batch_size # batch size used while training , i.e. no. of instances at time
        self.sFlat = None # strores the sFlat representation of midi file
        self.top = 0 #increases by every file
        self.trk = trk #which track (lead : 0, chorus : 1, drum : 2)
        self.DEPTH = constant.depths_of_3tracks[trk] # depth parrameter while making of sFlat...
        self.iter = 0 #Increases by every batch size
        self.ip_memory = ip_memory
        self.norm = norm
        self.flat_shape = None
        e = 32 if enc else 32
        self.shape = (batch_size, ip_memory, self.DEPTH)
        self.oshape = (batch_size, ip_memory, self.DEPTH * e)
        self.steps_per_epoch = steps_per_epoch
        self.steps = 0
        self.enc = enc # if to encode the sFlat to octave encoding
        if steps_per_epoch == 0:
            self.calc_steps_per_epoch()
            self.top = 0
        
        # print("steps per epochs  : ", self.steps_per_epoch)
        self.__read__()
        # self.on_epoch_end()

    def calc_steps_per_epoch(self):
        # i = 0
        # print("sequences : ", self.seq_names)
        for mid in self.seq_names:
            try :
                ns = dataset.to_note_sequence(mid)
            except:
                continue
            ns = dataset.merge_to_3tracks(ns)
            ar = dataset.ns_to_tarray(ns, resolution=64)
            if ar.shape[0] < 3: continue
            self.sFlat = dataset.ns_tarray_to_sFlat(t_arr= ar[ self.trk: self.trk + 1 ], DEPTH= self.DEPTH)
            self.steps_per_epoch += ((self.sFlat.shape[1] // self.batch_size)) - 1
                    

    def __next_file__(self):
        self.top += 1
        if self.top == len(self.seq_names) + 1 : return False
        # print("Top : ", self.top)
        return self.seq_names[self.top - 1]
    
    def __read__(self):
        ns = None
        if self.steps_per_epoch == 0: raise FileNotFoundError("Any of MIDI file in given dataset_path : "+ self.seq_names + " not reaable")
        while not self.__exit__():
            try  :
                filec = self.__next_file__()
                ns = dataset.to_note_sequence(filec)
                break
            except: 
                continue
        if not ns :return False
        ns = dataset.merge_to_3tracks(ns)     
        ar = dataset.ns_to_tarray(ns, resolution=64)
        # print("AR shape : ", ar.shape, self.trk)
        if ar.shape[0] <= self.trk : return self.__read__()
        # print("AGAIN AR shape : ", ar.shape)
        
        self.sFlat = dataset.ns_tarray_to_sFlat(t_arr= ar[ self.trk: self.trk + 1 ], DEPTH= self.DEPTH)
        self.sFlat = self.sFlat[:, : (self.sFlat.shape[1] // self.batch_size) * self.batch_size + 1] 
        self.steps = self.sFlat.shape[1] // self.batch_size  - 1
        self.iter = 0
        self.sFlat = self.sFlat if self.norm else sFlat
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
        if self.steps <= 0: self.__read__()
        enc = self.sFlat[:, self.iter : self.iter + self.batch_size + self.ip_memory]
        # print("sFlat shape : ", enc.shape)
        x, y = dataset.prepare_sFlat_data(enc, enc_shape= (self.DEPTH,), ip_memory=self.ip_memory, depth= self.DEPTH)
        # print("Prepare shape : ", x.shape, y.shape)
        # if self.enc: enc = enc_deco.sFlat_to_octave(self.sFlat[:, self.iter : self.iter + self.batch_size + self.ip_memory])  #Improving started 
        # shape = enc.shape[-2: ] if self.enc else tuple()
        x, y = x / 128, enc_deco.sFlat_to_octave(y)
        # print("Prepare Encoded shape : ", x.shape, y.shape)
        # print(shape, enc.shape)
        # print(x.shape, y.shape, '----> x, y', self.flat_shape)
        x, y = numpy.reshape(x, x.shape[1:3] + (-1, )), numpy.reshape(y, y.shape[1:2] + (-1, )) #reshaping to fit as rnn input
        # print("Prepare shape reshsape : ", x.shape, y.shape)
        self.iter += self.batch_size
        self.steps -= 1
        # print("steps : ", self.steps)
        # print(x.shape, y.shape, '----> x, y')
        return x, y

    def __str__(self):
        return '{\n\ttrk : ' + str(self.trk) + "\n\tseq_name : " + str(self.seq_names) + "\n\tbatch_size : " + str(self.batch_size) + \
                "\n\tshape : " + str(self.shape) + '\n\tsFlat_shape : ' + str(self.flat_shape) + '\n\tsteps_per_epochs : ' + str(self.steps_per_epoch) + \
                '\n\titer : ' + str(self.iter) +'\n\tEND\n}'


"""
It generates the note data batch wise
Returns:
    NoteDataGenerator -- generator class for note while trainin
"""
class NoteTimeDataGenerator(Sequence):
    def __init__(self, trk, seq_names, batch_size, ip_memory, enc = True, steps_per_epoch = 0, norm = True, bits = 8):
        self.seq_names = numpy.array(seq_names) # list of midi files avalable
        self.batch_size = batch_size # batch size used while training , i.e. no. of instances at time
        self.sFlat = None # strores the sFlat representation of midi file
        self.time = None
        self.top = 0 #increases by every file
        self.trk = trk #which track (lead : 0, chorus : 1, drum : 2)
        self.DEPTH = constant.depths_of_3tracks[trk] # depth parrameter while making of sFlat...
        self.iter = 0 #Increases by every batch size
        self.ip_memory = ip_memory
        self.norm = norm
        self.quanta = 2 #[0, 1, 2, ...., 32] out off 32, 32 means whole note
        self.iftime = False
        self.flat_shape = None
        self.bits = bits
        e = 32 if enc else 32
        self.shape = (batch_size, ip_memory, self.DEPTH * bits)
        self.oshape = (batch_size, ip_memory, self.DEPTH * e)
        self.steps_per_epoch = steps_per_epoch
        self.steps = 0
        self.enc = enc # if to encode the sFlat to octave encoding
        if steps_per_epoch == 0:
            self.calc_steps_per_epoch()
            self.top = 0

        self.__read__()
        # self.on_epoch_end()

    def calc_steps_per_epoch(self):
        for mid in self.seq_names:
            try :
                ns = dataset.to_note_sequence(mid)
            except:
                continue
            ns = dataset.merge_to_3tracks(ns)
            ar = dataset.ns_to_tarray(ns, resolution=64)
            if ar.shape[0] < 3: continue
            self.sFlat = d2.ns_tarray_to_sFlatroll(tarray= ar[ self.trk: self.trk + 1 ], quanta = self.quanta, depth= self.DEPTH)
            self.steps_per_epoch += ((self.sFlat.shape[1] // self.batch_size)) - 20
                    

    def __next_file__(self):
        self.top += 1
        if self.top == len(self.seq_names) + 1 : return False
        return self.seq_names[self.top - 1]
    
    def __read__(self):
        ns = None
        if self.steps_per_epoch == 0: raise FileNotFoundError("Any of MIDI file in given dataset_path : "+ self.seq_names + " not reaable")
        while not self.__exit__():
            try  :
                filec = self.__next_file__()
                ns = dataset.to_note_sequence(filec)
                break
            except: 
                continue
        if not ns :return False
        ns = dataset.merge_to_3tracks(ns)     
        ar = dataset.ns_to_tarray(ns, resolution=64)
        if ar.shape[0] <= self.trk : return self.__read__()
        
        # print("Updated the successfully")
        
        self.sFlat = d2.ns_tarray_to_sFlatroll(tarray= ar[ self.trk: self.trk + 1 ], quanta= self.quanta ,depth= self.DEPTH)
        self.sFlat = self.sFlat[:, : (self.sFlat.shape[1] // self.batch_size) * self.batch_size + 1] 
        self.time = dataset.ns_tarray_to_time(t_arr= ar[ self.trk: self.trk + 1 ])
        self.steps = self.sFlat.shape[1] // self.batch_size  - 1
        self.iter = 0
        self.sFlat = self.sFlat if self.norm else sFlat
        self.flat_shape = self.sFlat.shape
        # print("Updated the successfully")
        return True
    def __len__(self):
        return len(self.seq_names)

    def __exit__(self):
        if self.top == len(self.seq_names) : return True
        return False
    def on_epoch_end(self):
        self.top = 0

    def __getitem__(self, idx):
        if self.steps <= 0: self.__read__()
        if idx == 0: 
            # print("Reading the file ....", self.trk)
            self.__read__() #reading the new file atleast to get first smaple, to init everything , that is not understanding now
        enc = self.sFlat[:, self.iter : self.iter + self.batch_size + self.ip_memory]
        x, y = dataset.prepare_sFlat_data(enc, enc_shape= (self.DEPTH + 1,), ip_memory=self.ip_memory, depth= self.DEPTH )
        x, y = enc_deco.binary(x, self.bits) , enc_deco.sFlat_to_octave(y)
       
        x, y = numpy.reshape(x, x.shape[1:3] + (-1, )), numpy.reshape(y, y.shape[1:2] + (-1, )) #reshaping to fit as rnn input
        # print("x.shape, y.shape : ", x.shape, y.shape  )
        self.iter += self.batch_size
        self.steps -= 1
        
        return x, y

    def __str__(self):
        return '{\n\ttrk : ' + str(self.trk) + "\n\tseq_name : " + str(self.seq_names) + "\n\tbatch_size : " + str(self.batch_size) + \
                "\n\tshape : " + str(self.shape) + '\n\tsFlat_shape : ' + str(self.flat_shape) + '\n\tsteps_per_epochs : ' + str(self.steps_per_epoch) + \
                '\n\titer : ' + str(self.iter) +'\n\tEND\n}'


class DataGenerator_3Tracks(Sequence):
    def __init__(self, seq_names, batch_size, ip_memory, enc = True, steps_per_epoch = 0, norm = True, bits = 8, test_size = 0.2):
        self.ftrain, self.fval = train_test_split(seq_names, test_size= test_size)
        print("len : train, val : ", len(self.ftrain), len(self.fval))
        self.train = sFlatDataGenerator_3Tracks(self.ftrain, batch_size, ip_memory, enc, steps_per_epoch, norm)
        self.val = sFlatDataGenerator_3Tracks(self.fval, batch_size, ip_memory, enc, steps_per_epoch, norm, bits)
        self.ip_memory = ip_memory
        self.batch_size = batch_size
        self.bits = bits
    def __len__(self):
        return len(self.seq_names)
    pass

class sFlatDataGenerator_3Tracks(Sequence):
    def __init__(self, seq_names, batch_size, ip_memory, enc = True, steps_per_epoch = 0, norm = True, bits = 8):
        self.ip_memory = ip_memory
        self.batch_size = batch_size
        self.bits = bits 
        self.lead = NoteTimeDataGenerator(0, seq_names, batch_size, ip_memory, enc, steps_per_epoch, norm)
        self.chorus = NoteTimeDataGenerator(1, seq_names, batch_size, ip_memory, enc, steps_per_epoch, norm)
        self.drum = NoteTimeDataGenerator(2, seq_names, batch_size, ip_memory, enc, steps_per_epoch, norm)
        self.iter = -1
        self.steps_per_epoch = min([self.lead.steps_per_epoch, self.chorus.steps_per_epoch, self.drum.steps_per_epoch])

    def __len__(self):
        return self.steps_per_epoch
    def on_epoch_end(self):
        self.lead.on_epoch_end()
        self.chorus.on_epoch_end()
        self.drum.on_epoch_end()

        self.lead.top = 0
        self.chorus.top = 0
        self.drum.top = 0

        self.iter = -1

    def __getitem__(self, idx):
        self.iter += 1
        x0, y0 = self.lead.__getitem__(self.iter) 
        x1, y1 = self.chorus.__getitem__(self.iter)
        x2, y2 = self.drum.__getitem__(self.iter)
        return [x0, x1, x2], [y0, y1, y2]

    

def note_data(f, trk = 0, idx = None, ip_memory = 32, batch_size= 32, DEPTH = 1, all_ = False, randm = True):
    # following reads the file to sFalt representaion
    ns = dataset.to_note_sequence(f)
    ar = dataset.ns_to_tarray(ns, resolution= 64)
    sFlat = dataset.ns_tarray_to_sFlat(t_arr= ar[trk: trk + 1 ], DEPTH= DEPTH)

    MX = (sFlat.shape[1] - ip_memory - batch_size)
    if MX < 0: MX = 1
    idx = idx if idx else random.randint(0, MX) # get index which slice of ip_memory you want
    if idx > MX: raise Exception("Index out of bound err : Not in midi file") # if index is greater than MX, out of file 
    x, y = dataset.prepare_sFlat_data(sFlat[:, idx : idx + batch_size + ip_memory], ip_memory=ip_memory, depth= DEPTH)
    y = enc_deco.sFlat_to_octave(y)  # Improving started 
    if all_ : return x[0], y[0]
    rx = random.randint(0, x.shape[1])
    print("random init : ", rx)
    if randm: x[0, rx], y[0, rx]
    return x[0, 0], y[0, 0]
        


"""
Time data generator for batch_training

Returns:
    [type] -- [description]
"""
class TimeDataGenerator(Sequence):
    def __init__(self, trk, seq_names, batch_size, ip_memory):
        self.seq_names = numpy.array(seq_names) # list of midi files avalable
        self.batch_size = batch_size # batch size used while training , i.e. no. of instances at time
        self.sFlat = None # strores the time instanses for sFlat representation of midi file
        self.top = 0 #increases by every file
        self.trk = trk #which track (lead : 0, chorus : 1, drum : 2)
        # self.DEPTH = constant.depths_of_3tracks[trk] # depth parrameter while making of sFlat...
        self.iter = 0 #Increases by every batch size
        self.ip_memory = ip_memory
        self.flat_shape = None
        self.shape = (batch_size, ip_memory, 64)
        self.steps_per_epoch = 0
        self.steps = 0
        
        self.calc_steps_per_epoch()
        self.top = 0
        self.__read__()
        # self.on_epoch_end()

    def calc_steps_per_epoch(self):
        for mid in self.seq_names:
            filec = self.__next_file__()
            try : 
                ns = dataset.to_note_sequence(filec)
            except: continue
            ar = dataset.ns_to_tarray(ns, resolution=64)
            self.sFlat = dataset.ns_tarray_to_time(t_arr= ar[ self.trk: self.trk + 1 ])
            self.steps_per_epoch += ((self.sFlat.shape[1] // self.batch_size) * self.batch_size + 1) // self.batch_size  - 1
                    

    def __next_file__(self):
        self.top += 1
        if self.top == len(self.seq_names) : return False
        return self.seq_names[self.top - 1]
    
    def __read__(self):
        while not self.__exit__():
            try  :
                filec = self.__next_file__()
                ns = dataset.to_note_sequence(filec)
                break
            except: continue
                
        ar = dataset.ns_to_tarray(ns, resolution=64)
        self.sFlat = dataset.ns_tarray_to_time(t_arr= ar[ self.trk: self.trk + 1 ])
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
        if self.steps <= 0: self.__read__()
        
        enc = enc_deco.tm_to_enc_tm(self.sFlat[:, self.iter : self.iter + self.batch_size + self.ip_memory])  #None 
        x, y = dataset.prepare_sFlat_data(enc, enc_shape= enc.shape[-2: ], ip_memory=self.ip_memory, depth= self.DEPTH)
        print(x.shape, y.shape, '----> x, y', self.flat_shape)
        # x, y = numpy.reshape(x, x.shape[1:3] + (-1, )), numpy.reshape(y, y.shape[1:2] + (-1, )) #reshaping to fit as rnn input
        self.iter += self.batch_size
        self.steps -= 1
        # print("steps : ", self.steps)
        # print(x.shape, y.shape, '----> x, y')
        return x, y

    def __str__(self):
        return '{\n\ttrk : ' + str(self.trk) + "\n\tseq_name : " + str(self.seq_names) + "\n\tbatch_size : " + str(self.batch_size) + \
                "\n\tshape : " + str(self.shape) + '\n\tsFlat_shape : ' + str(self.flat_shape) + '\n\tsteps_per_epochs : ' + str(self.steps_per_epoch) + \
                '\n\titer : ' + str(self.iter) +'\n\tEND\n}'
        