

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
from keras import backend as kback
import tensorflow as tf

from polymuse import dataset2 as d2 , constant
from polymuse.losses import rmsecat


from keras.losses import categorical_crossentropy, mean_squared_error

from numpy import random
ra = random.randint(0, 100)
with open("ra.txt", 'a+') as f:
    f.write( ( "\n" + str(ra)) )

random.seed(ra)

tf.set_random_seed(ra)

HOME = os.getcwd()



# import tensorflow as tf

# from keras import backend as kback


def load(model): #loads model from .h5 file
    if type(model) == str: return load_model(model)

# def rmsecat(depth):
#     # print(y_true.shape, y_pred.shape, "=======================================================")
#     def rmsecat_(y_true, y_pred):
#         a = []
#         h_ = None
#         for i in range(depth * 2):
#             h__ = categorical_crossentropy(y_true[:, i : i + 16], y_pred[ :, i : i + 16]) 
#             # f_ = h__.eval(session= tf.compat.v1.Session())
#             # print(kback.eval(tf.shape(h__)), " o ooooooooooooooooooooooooo")
#             if h_ is None: h_ = tf.square(h__)
#             else: h_ += tf.square(h__)

#         print("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[----]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]   : ")
#         a = (tf.sqrt(h_) / (2 * depth))
#         print(tf.shape(a), "..............................................................")
#         # m = numpy.mean(a, axis= 1)
#         return a
#     def rmsecat__(y_true, y_pred):
#         e = 1e-12
#         y_pred = kback.clip(y_pred, e, 1. - e)
        

#     return rmsecat_

def rmsecat(depth):   
    def rmsecat_(y_true, y_pred):
        h_ = None
        for i in range(depth * 2):
            h__ = categorical_crossentropy(y_true[:, i : i + 16], y_pred[ :, i : i + 16]) 
            if h_ is None: h_ = tf.square(h__)
            else: h_ += tf.square(h__)
        a = (tf.sqrt(h_) / (2 * depth))
        return a
    return rmsecat_

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

def build_sFlat_model(data_gen, typ = 'piano',model_name = '000', IP = None, OP = None, cell_count = 256, epochs = 200, batch_size = 32, dropout = .3, dev = False, shuffle= False ):
    print("Shape : ", data_gen.shape)
    model = Sequential()
    # model.add(TimeDistributed(Flatten(input_shape=IP[1:])))

    model.add(CuDNNLSTM(cell_count, return_sequences=True, input_shape=data_gen.shape[1:]))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=True))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(cell_count))

    model.add(Dense(cell_count))
    
    model.add(Dense(data_gen.oshape[2]))

    model.add(Activation('softmax'))

    es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=50)

    # model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    adam = Adam(lr=0.0013)
    model.compile(loss=rmsecat(data_gen.DEPTH), optimizer=adam, metrics=['acc'])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    
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
    
    callbacks = callbacks_list
    history = model.fit_generator(data_gen, verbose = 1, steps_per_epoch = data_gen.steps_per_epoch, epochs= epochs,  callbacks= callbacks_list, shuffle = shuffle)
    j = 0

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

    if dev:
        F = 'gst' + file_name + '_t' + str(typ) + '_c' + str(cell_count) + '_e' + str(epochs) + '_d' + str(dropout) + '_b' + str(batch_size)
        if not os.path.exists('archive'): os.mkdir('archive')
        os.chdir('archive') 
        if not os.path.exists('hist'): os.mkdir('hist')
        os.chdir('hist')
        # if not os.path.exists('stateless'): os.mkdir('stateless')
        # os.chdir('stateless')
        with open( 'gst'+ file_name + '.json', 'w') as json_file:
            json.dump(history.history, json_file)
        os.chdir(HOME)

        if not os.path.exists('archive'): os.mkdir('archive')
        os.chdir('archive')
        if not os.path.exists('models'): os.mkdir('models')
        os.chdir('models')
        
        model.save(F + '.h5')

    os.chdir(HOME)
    with open('ra.txt', 'a+') as f:
        f.write("\t\tSuccess\n")
    return model


def build_sFlat_stateful_model(data_gen, typ = 'piano',model_name = '000', IP = None, OP = None, cell_count = 256, epochs = 200, batch_size = 32, dropout = .3, dev = False, shuffle = False):
    
    model = Sequential()
    # model.add(TimeDistributed(Flatten(input_shape=IP[1:])))

    batch_input_shape = data_gen.shape

    print("batch_input_shape : ", batch_input_shape)

    model.add(CuDNNLSTM(cell_count, return_sequences=True,  stateful=True, batch_input_shape = batch_input_shape, input_shape=data_gen.shape[1:]))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=True, batch_input_shape = batch_input_shape))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=True, batch_input_shape = batch_input_shape))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(data_gen.oshape[2]))

    model.add(Activation('softmax'))

    es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=50)

    # model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    model.compile(loss=rmsecat(data_gen.DEPTH), optimizer='adam', metrics=['acc'])
    
    # file_info = 'gsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"
    file_name  = ''.join(random.choice(list(string.ascii_lowercase)) for i in range(4))

    print('file_name : ', file_name)

    # Make the dir structure for saving h5Models
    if not os.path.exists('h5_models'): os.mkdir('h5_models')
    os.chdir('h5_models') 
    if not os.path.exists(constant.model_dict[typ]): os.mkdir(constant.model_dict[typ])
    os.chdir(constant.model_dict[typ])
    if not os.path.exists('stateful'): os.mkdir('stateful')
    os.chdir('stateful')

    checkpoint = ModelCheckpoint(
        file_name + '.h5', monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]
    
    callbacks = callbacks_list
    history = model.fit_generator(data_gen, verbose = 1, steps_per_epoch = data_gen.steps_per_epoch, epochs= epochs,  callbacks= callbacks_list, shuffle = shuffle)
    j = 0

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
    if not os.path.exists('stateful'): os.mkdir('stateful')
    os.chdir('stateful')

    
        

    with open(file_name + '.json', 'w') as json_file:
        json.dump(history.history, json_file)
    os.chdir(HOME)

    if dev:
        F = 'gst' + file_name + '_t' + str(typ) + '_c' + str(cell_count) + '_e' + str(epochs) + '_d' + str(dropout) + '_b' + str(batch_size)
        if not os.path.exists('archive'): os.mkdir('archive')
        os.chdir('archive') 
        if not os.path.exists('hist'): os.mkdir('hist')
        os.chdir('hist')
        # if not os.path.exists('stateless'): os.mkdir('stateless')
        # os.chdir('stateless')
        with open( 'gst'+ file_name + '.json', 'w') as json_file:
            json.dump(history.history, json_file)
        os.chdir(HOME)

        if not os.path.exists('archive'): os.mkdir('archive')
        os.chdir('archive')
        if not os.path.exists('models'): os.mkdir('models')
        os.chdir('models')
        
        model.save(F + '.h5')

    os.chdir(HOME)
    with open('ra.txt', 'a+') as f:
        f.write("\t\tSuccess\n")
    return model


from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten, CuDNNLSTM, TimeDistributed, Embedding
from polymuse import constant
from keras.layers.merge import concatenate

def build_sflattimebin_funct(gens = None, cell_count = 256, epochs = 200, batch_size = 32, dropout = .3, dev = False, shuffle = False, train = True):
    
    #Lead
    lead = Input(shape=(gens.ip_memory, constant.depths_of_3tracks[0] * gens.bits))
    lead_lstm_0 = CuDNNLSTM(cell_count, return_sequences=True)(lead)
    # lead_lstm_1 = CuDNNLSTM(cell_count, return_sequences=True)(lead_lstm_0)
    # lead_lstm_2 = CuDNNLSTM(cell_count, return_sequences=False)(lead_lstm_1)
    # lead_dense_0 = TimeDistributed(Dense(512, activation='relu')(lead_lstm_0))

    #Chorus
    chorus = Input(shape=(gens.ip_memory, constant.depths_of_3tracks[1] * gens.bits))
    chorus_lstm_0 = CuDNNLSTM(cell_count, return_sequences=True)(chorus)
    # chorus_lstm_1 = CuDNNLSTM(cell_count, return_sequences=True)(chorus_lstm_0)
    # chorus_lstm_2 = CuDNNLSTM(cell_count, return_sequences=False)(chorus_lstm_1)
    # chorus_dense_0 = TimeDistributed(Dense(512, activation='relu')(chorus_lstm_0))


    # Drum input 
    drum = Input(shape=(gens.ip_memory, constant.depths_of_3tracks[2] * gens.bits))
    drum_lstm_0 = CuDNNLSTM(cell_count, return_sequences=True)(drum)
    # drum_lstm_1 = CuDNNLSTM(cell_count, return_sequences=True)(drum_lstm_0)
    # drum_lstm_2 = CuDNNLSTM(cell_count, return_sequences=False)(drum_lstm_1)
    # drum_dense_0 = TimeDistributed(Dense(256, activation='relu')(drum_lstm_0))

    #concatenate the layers
    # polymuse = concatenate([lead_dense_0, chorus_dense_0, drum_dense_0])
    polymuse = concatenate([lead_lstm_0, chorus_lstm_0, drum_lstm_0])

    # polymuse_em = Embedding(output_dim=256, input_dim=constant.depths_of_3tracks[2] * gens.bits, input_length=gens.ip_memory)(polymuse
    tm = TimeDistributed(Dense(256))(polymuse)
    #Densing the layers
    # polymuse_n 
    polymuse_0 = LSTM(256, activation='relu', return_sequences=True)(tm)
    polymuse_1 = LSTM(128, activation='relu', return_sequences= True)(polymuse_0)
    polymuse_2 = LSTM(128, activation='relu', return_sequences= False)(polymuse_1)



    #Divididing out 
    lead_dense_1 = Dense(128, activation='relu', name = "lead_dense_1")(polymuse_2)
    lead_dense_2 = Dense(64, activation='relu', name = "lead_dense_2")(lead_dense_1)
    lead_dense_3 = Dense(32, activation='relu', name = "lead_track")(lead_dense_2)

    
    chorus_dense_1 = Dense(256, activation='relu', name = "chorus_dense_1")(polymuse_2)
    chorus_dense_2 = Dense(128, activation='relu', name = "chorus_dense_2")(chorus_dense_1)
    chorus_dense_3 = Dense(96, activation='relu', name = "chorus_track")(chorus_dense_2)

    
    drum_dense_1 = Dense(128, activation='relu', name = "drum_dense_1")(polymuse_2)
    drum_dense_2 = Dense(96, activation='relu', name = "drum_dense_2")(drum_dense_1)
    drum_dense_3 = Dense(64, activation='relu', name = "drum_track")(drum_dense_2)
    
    model = Model(inputs = [lead, chorus, drum], outputs= [lead_dense_3, chorus_dense_3, drum_dense_3], name = "Polymuse")

    # print(model.summary())

    # plot_model(model, to_file='polymuse_design.png')

    if train: train_sflatroll(model, gens, cell_count, epochs, batch_size, dropout, dev, shuffle)

    pass

def train_sflatroll(model, gens = None, cell_count = 256, epochs = 200, batch_size = 32, dropout = .3, dev = False, shuffle = False ):
    
    losses = {
	    "lead_track": rmsecat(1),
	    "chorus_track": rmsecat(3),
        "drum_track" : rmsecat(2)
    }


    es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=50)

    # model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    model.compile(loss=losses, optimizer='adam', metrics=['acc'])
    
    # file_info = 'gsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"
    file_name  = ''.join(random.choice(list(string.ascii_lowercase)) for i in range(4))

    print('file_name : ', file_name)

    # Make the dir structure for saving h5Models
    if not os.path.exists('h5_models'): os.mkdir('h5_models')
    os.chdir('h5_models') 


    checkpoint = ModelCheckpoint(
        file_name + '.h5', monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]
    
    callbacks = callbacks_list
    history = model.fit_generator(gens.train, verbose = 1, steps_per_epoch = gens.train.steps_per_epoch, epochs= epochs,  callbacks= callbacks_list, shuffle = shuffle, validation_data=gens.val, validation_steps = gens.val.steps_per_epoch // 8)
    j = 0

    print("Training Succesfull ...")
    
    model.save(file_name + '.h5')

    print("history ", history, ', : ')
   
    os.chdir(HOME)
    #making directory structure for history stores
    if not os.path.exists('hist'): os.mkdir('hist')
    os.chdir('hist') 
    

    
        

    with open(file_name + '.json', 'w') as json_file:
        json.dump(history.history, json_file)
    os.chdir(HOME)

    if dev:
        F = 'gst' + file_name + '_t' + '_c' + str(cell_count) + '_e' + str(epochs) + '_d' + str(dropout) + '_b' + str(batch_size)
        if not os.path.exists('archive'): os.mkdir('archive')
        os.chdir('archive') 
        if not os.path.exists('hist'): os.mkdir('hist')
        os.chdir('hist')
        # if not os.path.exists('stateless'): os.mkdir('stateless')
        # os.chdir('stateless')
        with open( 'gst'+ file_name + '.json', 'w') as json_file:
            json.dump(history.history, json_file)
        os.chdir(HOME)

        if not os.path.exists('archive'): os.mkdir('archive')
        os.chdir('archive')
        if not os.path.exists('models'): os.mkdir('models')
        os.chdir('models')
        
        model.save(F + '.h5')

    os.chdir(HOME)
    with open('ra.txt', 'a+') as f:
        f.write("\t\tSuccess\n")
    return model


    pass


def predict_w(model, ini, opshape = (2, 16)):
    # IP0, IP1, IP2 = ini[0].shape, ini[1].shape, ini[2].shape
    # sh1, sh2, sh3 = (1, IP0[2]) + opshape, (1, IP1[2]) + opshape, (1, IP2[2]) + opshape
    # x1, x2, x3 = ini[0].reshape(IP0[0], IP0[1], -1), ini[1].reshape(IP1[0], IP1[1], -1), ini[2].reshape(IP2[0], IP2[1], -1)
    y = model.predict_on_batch(ini)
    # print(y, list(v.shape for v in y))
    return y 
    pass

