from polymuse import dataset, transformer, enc_deco, dutils, dataset2 as d2, evaluation, constant, data_generator, rnn_h, rnn_gpu, pattern

import random, argparse


"""
The fuction is primarily used for traing the midis, from dataset_path
Note : training will done on cpu, which may take long time to train even few files

Raises:
    FileNotFoundError: If no midi found in specified dataset_path
"""
def train(dataset_path, maxx = 1, trks = [0, 1, 2], graph = False, epochs=50, dev = True, cell_count= 512):
    f = load_dataset()

    # print(f)    
    for i in trks:
        data_gen = data_generator.NoteDataGenerator(i, f, 32, 32) #init the data generator, this makes possible to train on large dataset which cannot fit into memory
        print("1 : ", constant.type3tracks[i])
        
        rnn_h.build_sFlat_model(data_gen, typ = constant.type3tracks[i], epochs=50, dev = True, cell_count= 512) # makes a models and train on Note data, sFlat representaion of midi, and then to octave encoding used for training 
    if graph : 
        #need to plot graph after traing each model. The all information regarding epochs is store in hist/model_type/model_nature/
        #hist store in json format, human redable format
        
        # drawer.draw_json_loss_acc(fn1, fn2)
        pass


def train_gpu(dataset_path, maxx = 1, trks = [0, 1, 2], graph = False, epochs=50, dev = True, cell_count= 512):
    f = load_dataset(dataset_path, maxx)
    # print(f)    
    for i in trks:
        data_gen = data_generator.NoteDataGenerator(i, f, 32, 32, enc= True) #init the data generator, this makes possible to train on large dataset which cannot fit into memory
        print("1 : ", constant.type3tracks[i])
        
        rnn_gpu.build_sFlat_model(data_gen, typ = constant.type3tracks[i], epochs= epochs, dev= dev, cell_count= cell_count) # makes a models and train on Note data, sFlat representaion of midi, and then to octave encoding used for training 
    if graph : 
        #need to plot graph after traing each model. The all information regarding epochs is store in hist/model_type/model_nature/
        #hist store in json format, human redable format
        
        # drawer.draw_json_loss_acc(fn1, fn2)
        pass

def train_stateful_gpu(dataset_path, maxx = 1, trks = [0, 1, 2], graph = False, epochs=50, dev = True, cell_count= 512):
    f = load_dataset(dataset_path, maxx)
    # print(f)    
    for i in trks:
        data_gen = data_generator.NoteDataGenerator(i, f, 32, 32) #init the data generator, this makes possible to train on large dataset which cannot fit into memory
        print("1 : ", constant.type3tracks[i])
        print("data_gen : ", data_gen)
        
        rnn_gpu.build_sFlat_stateful_model(data_gen, typ = constant.type3tracks[i], epochs= epochs, dev= dev, cell_count= cell_count) # makes a models and train on Note data, sFlat representaion of midi, and then to octave encoding used for training 
    if graph : 
        #need to plot graph after traing each model. The all information regarding epochs is store in hist/model_type/model_nature/
        #hist store in json format, human redable format
        
        # drawer.draw_json_loss_acc(fn1, fn2)
        pass
def train_single_track(dataset_path, maxx = 1, trk = 0, graph = False, epochs=50, dev = True, cell_count= 512):
    train_gpu(dataset_path, maxx= maxx, trks = [trk], graph= graph, epochs= epochs, dev= dev, cell_count= cell_count)


def load_dataset(dataset_path, maxx = 10):
    fs = dutils.get_all_midis(dataset_path, maxx= maxx) #get all the midi files path in dataset
    
    if fs == [] : raise FileNotFoundError("No MIDI file in given dataset_path : ", dataset_path)
    
    f = [] #store to store random midi from total midi files
    for i in range(maxx): f.append(random.choice(fs))

    if not maxx: f = fs # if maxx not specified train on all files in dataset
    return f
