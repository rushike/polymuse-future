

"""
Model building using keras library
This includes code for building the various RNN base models for our polymuse 
The models are store in current directory, in hierarchy
--h5_models/
    --piano/
        --stateful/
        --stateless/
    --lead/
        --stateful/
        --stateless/
    --drum/
        --stateful/
        --stateless/
    --chorus/
        --stateful/
        --stateless/
    --dense3/
        --stateful/
        --stateless/

The model includes the core functionality function as 
load --> to load h5 models from .h5 files
octave_loss --> to calculate the octave loss, not tested  
predict, predict_b, predict_dense --> predict the next note, time instance based on models and the current input

"""


import  numpy, datetime, json, os, string

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Activation, CuDNNLSTM, TimeDistributed, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adagrad, Adam, RMSprop
import tensorflow as tf

from polymuse import dataset2 as d2 , constant
from keras import backend as kback

from numpy import random
random.seed(131)
tf.set_random_seed(131)

HOME = os.getcwd()



# import tensorflow as tf

# from keras import backend as kback


def load(model): #loads model from .h5 file
    if type(model) == str: return load_model(model)

def predict(model, x, batch_size = 32):
    IP = x.shape
    sh = (IP[0], ) + x.shape[2:]
    x = x.reshape(IP[0], IP[1], -1)
    y = model.predict(x, verbose = 0)
    y = y.reshape(sh)
    return y 


def predict_b(model, x):
    IP = x.shape
    sh = (1, ) + x.shape[2:]
    x = x.reshape(IP[0], IP[1], -1)
    # print("x shape : ", x.shape)
    y = model.predict_on_batch(x)
    y = y.reshape(sh)
    return y 


def predict_dense(model, x):
    IP = x.shape
    sh = (1, ) + IP[1:]
    x = x.reshape((1, -1))
    # print(x.shape, "--x ; ;;;")
    y = model.predict_on_batch(x)
    y = y.reshape(sh)
    return y

def load_piano_drum_dense_models():
    drm_dense = HOME + '/h5_models/'
    home = HOME + '/h5_models/'
    dirs= ['lead', 'drum', 'dense3\\dense']
    op_models = []
    
    for i, p in enumerate(dirs):
        models = os.listdir(home + p)
        print(p, models)
        if models[0].startswith("gsF"):
            mv = load_model(home + p + "\\" + models[0])
            mt = load_model(home + p + "\\" + models[1])
        elif models[0].startswith("gTsF"):
            mt = load_model(home + p + "\\" + models[0])
            mv = load_model(home + p + "\\" + models[1])
        elif models[0].startswith("gDnDF_1"):
            mv = load_model(home + p + "\\" + models[0])
            mt = load_model(home + p + "\\" + models[1])
        elif models[0].startswith("gDnDF_2"):
            mv = load_model(home + p + "\\" + models[1])
            mt = load_model(home + p + "\\" + models[0])



        op_models.append(mv)
        op_models.append(mt)
        # models = [home + p + "\\" + m for m in models]
        # print(models)
    return tuple(op_models)

def build_sFlat_model(data_gen, typ = 'piano',model_name = '000', IP = None, OP = None, cell_count = 256, epochs = 200, batch_size = 32, dropout = .3 ):

    model = Sequential()
    # model.add(TimeDistributed(Flatten(input_shape=IP[1:])))

    model.add(CuDNNLSTM(cell_count, return_sequences=True, input_shape=data_gen.shape[1:]))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=True))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=True))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(data_gen.shape[2]))

    model.add(Activation('softmax'))

    es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=50)

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    
    # file_info = 'gsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"
    file_name  = ''.join(random.choice(list(string.ascii_lowercase)) for i in range(4))

    print('file_name : ', file_name)

    # Make the dir structure for saving h5Models
    if not os.path.exists('h5_models'): os.mkdir('h5_models')
    os.chdir('h5_models') 
    if not os.path.exists(constant.model_dict[typ]): os.mkdir(constant.model_dict[typ])
    os.chdir(constant.model_dict[typ])
    if not os.path.exists('stateless'): os.mkdir('stateless')
    os.chdir('stateless')

    checkpoint = ModelCheckpoint(
        file_name + '.h5', monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]
    
    
    history = model.fit_generator(data_gen, verbose = 1, steps_per_epoch = data_gen.steps_per_epoch, epochs= epochs, callbacks = callbacks_list,  shuffle = False)
    j = 0
    # while True:
    #     x, y = data_gen.__getitem__(j)
    #     history = model.train_on_batch(x , y)
    #     if data_gen.__exit__(): break

    print("Training Succesfull ...")
    
    model.save(file_name + '.h5')

    print("history ", history, ', : ')
   
    # f = HOME + '/hist/piano/stateless/g_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
    os.chdir(HOME)
    #making directory structure for history stores
    if not os.path.exists('hist'): os.mkdir('hist')
    os.chdir('hist') 
    if not os.path.exists(constant.model_dict[typ]): os.mkdir(constant.model_dict[typ])
    os.chdir(constant.model_dict[typ])
    if not os.path.exists('stateless'): os.mkdir('stateless')
    os.chdir('stateless')


    with open(file_name + '.json', 'w') as json_file:
        json.dump(history.history, json_file)
    os.chdir(HOME)
    return model
