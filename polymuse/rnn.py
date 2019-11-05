





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






import  numpy, datetime, json, os

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Activation, CuDNNLSTM, TimeDistributed, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adagrad, Adam, RMSprop
import tensorflow as tf

from polymuse import dataset2 as d2 
from polymuse.losses import rmsecat
from keras import backend as kback

from numpy import random
random.seed(131)
tf.set_random_seed(131)

HOME = os.getcwd()



def load(model): #loads model from .h5 file
    if type(model) == str: return load_model(model)


def octave_loss(y_true, y_pred):
    # y_true, y_pred = tf.keras.utils.normalize(y_true, 1), tf.keras.utils.normalize(y_pred, 1)
    mxt, mxp = kback.max(y_true, 1), kback.max(y_pred, 1)
    mnt, mnp = kback.min(y_true, 1), kback.min(y_pred, 1)
    # y_true, y_pred = (y_true - mnt) / (mxt - mnt), (y_pred - mnp)/(mxp - mnp)
    print(y_true[0], y_pred[0])
    print("y_true : ", y_true.shape, " y_pred : ", y_pred.shape)
    octv_t, note_t = tf.argmax(y_true[:, :16], 1), tf.argmax(y_true[:, 16:], 1)
    octv_p, note_p = tf.argmax(y_pred[:, :16], 1), tf.argmax(y_pred[:, 16:], 1) 
    n_t = (octv_t * 12 + note_t) / 128
    n_p = (octv_p * 12 + note_p) / 128

    n_t = tf.cast(n_t, tf.float32)
    
    n_p = tf.cast(n_p, tf.float32)

    print(n_t, n_p)
    print("nt_shape : ", n_t.shape, ", np_shape : ", n_p.shape)

    diff = tf.abs(n_t - n_p) / kback.max(n_t)

    print(diff)
    print("diff : ", diff.shape)

    fact = abs(diff - 7 )// 7
    fact2 = abs(diff - 7) / 7

    print(fact)
    print("fact : ", fact.shape)

    print("==================================")
    print(fact2)
    print("fact2__ : ", fact2.shape)

    # oct_loss = (abs(octv_t - octv_p) * fact * fact2 + tf.square(y_pred[:16]) - 1 + fact2) / 1.0
    # oct_loss = tf.abs(tf.math.subtract(n_p[:16], n_t[:16])) / 300 + tf.square(y_pred[:, :16]) - 1
    n_loss = kback.square(y_pred[:, 16:])
    oct_loss = tf.abs((tf.abs(n_p - n_t) * fact) + kback.mean(n_loss)  - 1 + tf.square(diff))

    print(oct_loss, oct_loss.shape)
    return oct_loss 


def rmsecat(depth):   
    def rmsecat_(y_true, y_pred):
        a = []
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


def build_sFlat_model(x, y, model_name, IP = None, OP = None, cell_count = 256, epochs = 200, batch_size = 32, dropout = .3 ):
    # LE = x.shape[0] //batch_size

    # x, y = x[:LE * batch_size], y[:LE * batch_size]

    ip_memory = x.shape[1]
    
    IP = x.shape if not IP else IP
    OP = y.shape if not OP else OP

    x = x.reshape(IP[0], IP[1], -1)
    y = y.reshape(OP[0], -1)
    
    IP = x.shape if not IP else IP
    OP = y.shape if not OP else OP

    print("IP: ", IP, x.shape)
    print("OP: ", OP, y.shape)
    model = Sequential()
    # model.add(TimeDistributed(Flatten(input_shape=IP[1:])))
    
    batch_input_shape = (batch_size,) + x.shape[1:]
    print("batch input shape : ", batch_input_shape)
    model.add(CuDNNLSTM(cell_count, return_sequences=True, input_shape=(IP[1], numpy.prod(IP[2:]))))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=True))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=True))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=False))
    model.add(Dropout(dropout))

    # model.add(CuDNNLSTM(cell_count, return_sequences=False))
    # model.add(Dense(cell_count // 2))
    # model.add(Dropout(dropout))
    
    model.add(Dense(numpy.prod(IP[2:])))

    model.add(Activation('softmax'))

    es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=50)

    # model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['acc'])
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    # model.compile(loss=octave_loss, optimizer='rmsprop', metrics=['acc', octave_loss])
    # batch_size = 30 if not batch_size else batch_size
    # epochs = 500 if not epochs else epochs
    file_path = HOME + '/h5_models/piano/stateless/gsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"
    

    checkpoint = ModelCheckpoint(
        file_path, monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]
    
    
    history = model.fit(x, y, validation_split = 0.2,  nb_epoch=epochs, callbacks = callbacks_list, shuffle = False)
    print("history keys : " , history.history.keys())
    
    model.save(file_path)
   
    f = HOME + '/hist/piano/stateless/g_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
    
    with open(f, 'w') as json_file:
        json.dump(history.history, json_file)

    return model


def build_sFlat_stateful_model(x, y, model_name, IP = None, OP = None, cell_count = 256, epochs = 200, batch_size = 32, dropout = .3 ):
    LE = x.shape[0] //batch_size

    x, y = x[:LE * batch_size], y[:LE * batch_size]

    ip_memory = x.shape[1]
    
    IP = x.shape if not IP else IP
    OP = y.shape if not OP else OP

    x = x.reshape(IP[0], IP[1], -1)
    y = y.reshape(OP[0], -1)
    
    IP = x.shape if not IP else IP
    OP = y.shape if not OP else OP

    print("IP: ", IP, x.shape)
    print("OP: ", OP, y.shape)
    model = Sequential()
    # model.add(TimeDistributed(Flatten(input_shape=IP[1:])))
    
    batch_input_shape = (batch_size,) + x.shape[1:]
    print("batch input shape : ", batch_input_shape)
    model.add(CuDNNLSTM(cell_count, return_sequences=True, stateful=True, batch_input_shape = batch_input_shape, input_shape=(IP[1], numpy.prod(IP[2:]))))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=True, stateful=True))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=True, stateful=True,))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=False))
    model.add(Dropout(dropout))

    # model.add(CuDNNLSTM(cell_count, return_sequences=False))
    # model.add(Dense(cell_count // 2))
    # model.add(Dropout(dropout))
    
    model.add(Dense(numpy.prod(IP[2:])))

    model.add(Activation('softmax'))

    es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=50)

    # model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['acc'])
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    # model.compile(loss=octave_loss, optimizer='rmsprop', metrics=['acc', octave_loss])
    # batch_size = 30 if not batch_size else batch_size
    # epochs = 500 if not epochs else epochs
    file_path = HOME + '/h5_models/piano/stateful/gsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"
    

    checkpoint = ModelCheckpoint(
        file_path, monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]
    
    
    history = model.fit(x, y, validation_split = 4 * batch_size/ x.shape[0],  nb_epoch=epochs, callbacks = callbacks_list,  shuffle = False)
    print("history keys : " , history.history.keys())
    
    model.save(file_path)
   
    f = HOME + '/hist/piano/stateful/g_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
    
    with open(f, 'w') as json_file:
        json.dump(history.history, json_file)

    return model


def build_time_sFlat_stateful_model(x, y, model_name, IP = None, OP = None, cell_count = 256, epochs = 200, batch_size = 25, dropout = .5):
    LE = x.shape[0] //batch_size

    x, y = x[:LE * batch_size], y[:LE * batch_size]
    ip_memory = x.shape[1]
    
    IP = x.shape if not IP else IP
    OP = y.shape if not OP else OP

    x = x.reshape(IP[0], IP[1], -1)
    y = y.reshape(OP[0], -1)
    
    IP = x.shape if not IP else IP
    OP = y.shape if not OP else OP

    print("IP: ", IP)
    print("OP: ", OP)
    model = Sequential()
    batch_input_shape = (batch_size,) + x.shape[1:]
    print("batch input shape : ", batch_input_shape)
    model.add(CuDNNLSTM(cell_count, return_sequences=True, stateful=True, batch_input_shape = batch_input_shape,input_shape=(IP[1], numpy.prod(IP[2:]))))
    # model.add(Dense(cell_count))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=True, stateful=True))
    # model.add(Dense(cell_count // 2))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=False))
    # model.add(Dense(cell_count // 2))
    model.add(Dropout(dropout))
    
    model.add(Dense(numpy.prod(IP[2:])))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    # batch_size = 30 if not batch_size else batch_size
    # epochs = 500 if not epochs else epochs
 
    file_path = HOME + '/h5_models/piano/stateful/gTsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"
    
    es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=50)

    checkpoint = ModelCheckpoint(
        file_path, monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]
    
    history = model.fit(x, y, validation_split =  4 * batch_size / x.shape[0], shuffle = False,batch_size=batch_size,  nb_epoch=epochs, callbacks = callbacks_list)
    

    
    model.save(file_path)

    f = HOME + '/hist/piano/stateful/gTsF_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
    with open(f, 'w') as json_file:
        json.dump(history.history, json_file)
    1 == 0
    return model

    """
    REPORT
    If validation split is increase from 0.1 to 0.3 , the validation loss , and validation accuracy increase to 0.17 - 0.20 
    If cut out dense layer before dropout, in middle layer validation loss, accuracy  remains same , but loss while training was high 
    If increase the one more layer, and dense layer after LSTM than dropout, then there is no as such change but need to experiment more to get clear value. But Validation is not increaseing as such
    Everything above just made things nonssense, no learning

    """


def build_time_sFlat_model(x, y, model_name, IP = None, OP = None, cell_count = 256, epochs = 200, batch_size = 25, dropout = .5):
    # LE = x.shape[0] //batch_size

    # x, y = x[:LE * batch_size], y[:LE * batch_size]
    ip_memory = x.shape[1]
    
    IP = x.shape if not IP else IP
    OP = y.shape if not OP else OP

    x = x.reshape(IP[0], IP[1], -1)
    y = y.reshape(OP[0], -1)
    
    IP = x.shape if not IP else IP
    OP = y.shape if not OP else OP

    print("IP: ", IP)
    print("OP: ", OP)
    model = Sequential()
    batch_input_shape = (batch_size,) + x.shape[1:]
    print("batch input shape : ", batch_input_shape)
    model.add(CuDNNLSTM(cell_count, return_sequences=True, input_shape=(IP[1], numpy.prod(IP[2:]))))
    # model.add(Dense(cell_count))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=True))
    # model.add(Dense(cell_count // 2))
    model.add(Dropout(dropout))

    model.add(CuDNNLSTM(cell_count, return_sequences=False))
    # model.add(Dense(cell_count // 2))
    model.add(Dropout(dropout))
    
    model.add(Dense(numpy.prod(IP[2:])))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    # batch_size = 30 if not batch_size else batch_size
    # epochs = 500 if not epochs else epochs
 
    file_path = HOME + '/h5_models/piano/stateless/gTsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"
    
    es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=50)

    checkpoint = ModelCheckpoint(
        file_path, monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]
    
    history = model.fit(x, y, validation_split =  0.2, shuffle = False,batch_size=batch_size,  nb_epoch=epochs, callbacks = callbacks_list)
    

    
    model.save(file_path)

    f = HOME + '/hist/piano/stateless/gTsF_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
    with open(f, 'w') as json_file:
        json.dump(history.history, json_file)
    1 == 0
    return model


def evalulate(model, x, y):
    y1, y2 = numpy.zeros(y.shape[0]), numpy.zeros(y.shape[0])

    y_ = model.predict_on_batch(x)

    print(y_.shape)

    for i in range(y_.shape[0]):
        y1[i] = numpy.argmax(y[i])
        y2[i] = numpy.argmax(y_[i])
    return y1, y2



def drum_note_h_dense(note, drum, x_n, y_n, x_d, y_d, model_name,  IP = None, OP = None, dense_count = 96, epochs = 200, batch_size = 32, lr = 0.01):
    x_n_ = numpy.zeros(y_n.shape)
    x_d_ = numpy.zeros(y_d.shape)
    print("wait : loading : ")
    for i in range(x_n_.shape[0]):
        x_n_[i] = predict_b(note, x_n[i : i + 1])
        if i % 10 == 0:
            print(".", end="")
    print("\n========")
    
    for i in range(x_d_.shape[0]):
        x_d_[i] = predict_b(drum, x_d[i: i + 1])
        if i % 10 == 0: 
            print(".", end="")
    
    print("\n", x_n_.shape, x_d_.shape, ' --- x_n_,  x_d_ ')

    x= d2.merge_rolls(x_n_, x_d_, 1)
    y1 = y_d[:, 0]
    y2 = y_d[:, 1]

    print(x.shape, y1.shape, "-- x, y1 ")
    print(x.shape, y2.shape, "-- x, y2 ")

    print("Start training the for model one. .. ..  . .. .. .. ... ")
    model_one = drum_note_dense(x, y1, model_name, dense_count= 96, epochs= epochs, batch_size= batch_size, lr= lr, ser= 1)
    print("Start training the for model two - -- - - ---- -- -- -- ")
    model_two = drum_note_dense(x, y1, model_name, dense_count= 96, epochs= epochs, batch_size= batch_size, lr= lr, ser= 2)
    
    # IP_ = x.shape if not IP else IP
    # OP_ = y.shape if not OP else OP

    return model_one, model_two

def drum_note_dense(x, y, model_name,  IP = None, OP = None, dense_count = 96, epochs = 200, batch_size = 32, lr = 0.01, ser = 1):
    x, y = numpy.reshape(x, (x.shape[0], -1)), numpy.reshape(y, (y.shape[0], -1))
    IP, OP = x.shape, y.shape
    
    print("IP : ", IP)
    print("OP : ", OP)

    model  = Sequential()
    model.add(Dense(IP[1], input_dim = IP[1]))

    model.add(Dense(32 * IP[1], activation='softmax'))
    model.add(Dense(16 * IP[1], activation='softmax'))

    model.add(Dense(OP[1], activation='softmax'))

    ada = RMSprop(lr = lr)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    # batch_size = 30 if not batch_size else batch_size
    # epochs = 500 if not epochs else epochs
 
    file_path = HOME + '/h5_models/dense3/gDnDF_' + str(ser)+ "_" + str(dense_count) + "_lr_" +  str(lr) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + ".h5"
    
    es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=50)

    checkpoint = ModelCheckpoint(
        file_path, monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]
    
    history = model.fit(x, y, validation_split = 0.2, shuffle = False,batch_size=batch_size,  nb_epoch=epochs, callbacks = callbacks_list)
    
    model.save(file_path)

    # x, y = numpy.reshape(x, IP), numpy.reshape(y, OP)

    f = HOME + '/hist/dense3/gDnDF_h_' + str(dense_count) + "_lr_" +  str(lr) + '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + ".json"
    with open(f, 'w') as json_file:
        json.dump(history.history, json_file)
    1 == 0
    ac, acc = model.evaluate(x, y) 
    print("Model ", model_name, " trained with accuracy, prediction", ac, acc)
    return model


def drum_time_h(time, drum, x_t, y_t, x_d, y_d, model_name, IP = None, OP = None, dense_count = 96, epochs = 200, batch_size = 32):
    x_n_ = numpy.zeros(y_n.shape)
    x_d_ = numpy.zeros(y_d.shape)

    for i in range(x_n_.shape[0]):
        x_n_[i] = predict_b(time, x_n[i])
    
    for i in range(x_d_.shape[0]):
        x_d_[i] = predict_b(drum, x_d[i])
    
    print(x_d_.shape, x_d_.shape, ' --- x_n_,  x_d_ ')

    x, y = dutils.merge_roll(x_n_, x_d_), dutils.merge_roll(y_n, y_n)

    IP = x.shape if not IP else IP
    OP = y.shape if not OP else OP

    x, y = numpy.reshape(x, (IP_[0], -1)), numpy.reshape(y, (OP_[0], -1))

    IP, OP = x.shape, y.shape
    print("IP : ", IP)
    print("OP : ", OP)

    model  = Sequential()

    model.add(Dense(IP[1], input_dim = IP[1], activation='relu'))

    model.add(Dense(16 * IP[1], activation='relu'))
    model.add(Dense(16 * IP[1], activation='relu'))
    model.add(Dense(16 * IP[1], activation='relu'))

    model.add(Dense(OP[1], activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    # batch_size = 30 if not batch_size else batch_size
    # epochs = 500 if not epochs else epochs
 
    file_path = HOME + '/h5_models/gTtDF_' +  str(dense_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + ".h5"
    
    es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=50)

    checkpoint = ModelCheckpoint(
        file_path, monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint, es]
    
    history = model.fit(x, y,batch_size=batch_size,  nb_epoch=epochs, callbacks = callbacks_list)
    
    model.save(file_path)

    f = HOME + '/hist/gTtDF_h_' + str(dense_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + ".json"
    with open(f, 'w') as json_file:
        json.dump(history.history, json_file)
    1 == 0
    ac, acc = model.evaluate(x, y) 
    print("Model ", model_name, " trained with accuracy, prediction", ac, acc)
    return model




