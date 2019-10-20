
from polymuse import rnn, dutils, dataset, dataset2 as d2, enc_deco

import numpy
"""
rnn_player -- capable of playing/generating the music output as octave/time encoded representation

These also includes two most important functions :
    * shift:
    * add_flatroll: 
"""



def rsingle_note_time_stateful_play(model_note, model_time, ini_ip, ini_ip_tm, y_expected_note = None, y_expected_time = None, ip_memory = None, predict_instances = 250):
    model_note = rnn.load(model_note) if type(model_note) == str else model_note
    model_time = rnn.load(model_time) if type(model_time) == str else model_time
    
    ip_memory = ip_memory if ip_memory else ini_ip.shape[0]
    
    inp = numpy.array([ini_ip])
    inp = numpy.array(ini_ip)
    inp_tm = numpy.array(ini_ip_tm)
    print("inp time shape : ", inp_tm.shape)
    print('inp note shape : ', inp.shape)
    notes_shape = (1, predict_instances) + inp.shape[2:]
    time_shape = (1, predict_instances) + inp_tm.shape[2:]
    # notes_shape.extend(inp.shape[2:])
    # time_shape.extend(inp_tm.shape[2:])
    bs = inp.shape[0]

    predict_instances = (predict_instances // bs) * bs
    mem = inp.shape[1]
    
    notes = numpy.zeros(notes_shape)
    time = numpy.zeros(time_shape)
    print(bs, "--bs")
    print("notes, time : ", notes.shape, time.shape)
    # notes[0, :mem, :] = inp #initiating the start

    for tm in range(0, predict_instances, 32 ):
        # print("loop", tm)
        y = rnn.predict(model_note, inp)
        y_len = rnn.predict(model_time, inp_tm)
        # print(y.shape, " --------------")
        if 95 < tm < 150 :
            # print("inp tm : ", inp_tm)
            # print("time : ", time[0, :10])
            # print("shape : ", y.shape)
            # print("Expected y_len NOTE: ", y_expected_note[tm + 1])
            # print("y_len : ", dutils.arg_octave_max(y[0, 0]))
            
            # print("+=================================================================+")
            # print("Expected y_len : ", y_expected_time[tm + 1])
            # print("y_len --  : ", y_len[0])
            # print("y_len : ", dutils.arg_max(y_len[0]))
            # print("ynum argmax : ", numpy.argmax(y_len[0]))
            pass
        
        for j in range(bs):
            y_len[j] = dutils.arg_max(y_len[j])
        for j in range(bs):
            for i in range(y.shape[1]):
                y[j, i] = dutils.arg_octave_max(y[j, i])    

        notes[0, tm : tm + bs] = y
        time[0, tm : tm + bs] = y_len

        #Note Value
        inp = shift(inp, axis= 1)
        add_flatroll(inp, y)
        
        #Time Length
        inp_tm = shift(inp_tm, axis=1)
        add_flatroll(inp_tm, y_len)
        
        pass
    return notes, time

def rsingle_note_time_play(model_note, model_time, ini_ip, ini_ip_tm, y_expected_note = None, y_expected_time = None, ip_memory = None, predict_instances = 250):
    model_note = rnn.load(model_note) if type(model_note) == str else model_note
    model_time = rnn.load(model_time) if type(model_time) == str else model_time
    
    ip_memory = ip_memory if ip_memory else ini_ip.shape[0]
    
    inp = numpy.array([ini_ip])
    # inp = numpy.array(ini_ip)
    inp_tm = numpy.array([ini_ip_tm])
    print("inp time shape : ", inp_tm.shape)
    print('inp note shape : ', inp.shape)
    notes_shape = (1, predict_instances) + inp.shape[2:]
    time_shape = (1, predict_instances) + inp_tm.shape[2:]
    # notes_shape.extend(inp.shape[2:])
    # time_shape.extend(inp_tm.shape[2:])
    bs = inp.shape[0]

    predict_instances = (predict_instances // bs) * bs
    mem = inp.shape[1]
    
    notes = numpy.zeros(notes_shape)
    time = numpy.zeros(time_shape)
    print(bs, "--bs")
    print("notes, time : ", notes.shape, time.shape)
    # notes[0, :mem, :] = inp #initiating the start

    for tm in range(0, predict_instances):
        # print("loop", tm)
        y = rnn.predict_b(model_note, inp)
        y_len = rnn.predict_b(model_time, inp_tm)
        # print(y.shape, " --------------")
        if 95 < tm < 150 :
            # print("inp tm : ", inp_tm)
            # print("time : ", time[0, :10])
            # print("shape : ", y.shape)
            # print("Expected y_len NOTE: ", y_expected_note[tm + 1])
            # print("y_len : ", dutils.arg_octave_max(y[0, 0]))
            
            # print("+=================================================================+")
            # print("Expected y_len : ", y_expected_time[tm + 1])
            # print("y_len --  : ", y_len[0])
            # print("y_len : ", dutils.arg_max(y_len[0]))
            # print("ynum argmax : ", numpy.argmax(y_len[0]))
            pass
        
        for j in range(bs):
            y_len[j] = dutils.arg_max(y_len[j])
        for j in range(bs):
            for i in range(y.shape[1]):
                y[j, i] = dutils.arg_octave_max(y[j, i])    

        notes[0, tm : tm + bs] = y
        time[0, tm : tm + bs] = y_len

        #Note Value
        inp = shift(inp, axis= 1)
        add_flatroll(inp, y)
        
        #Time Length
        inp_tm = shift(inp_tm, axis=1)
        add_flatroll(inp_tm, y_len)
        
        pass
    return notes, time


def rsing_note_time_play(model_note, model_time, ini_ip, ini_ip_tm, y_expected_note = None, y_expected_time = None, ip_memory = None, predict_instances = 250):
    model_note = rnn.load(model_note) if type(model_note) == str else model_note
    model_time = rnn.load(model_time) if type(model_time) == str else model_time
    
    ip_memory = ip_memory if ip_memory else ini_ip.shape[0]

    inp = numpy.array([ini_ip])
    inp_tm = numpy.array([ini_ip_tm])
    print("inp time shape : ", inp_tm.shape)
    print('inp note shape : ', inp.shape)
    notes_shape = (1, predict_instances) + inp.shape[2:]
    time_shape = (1, predict_instances) + inp_tm.shape[2:]
    # notes_shape.extend(inp.shape[2:])
    # time_shape.extend(inp_tm.shape[2:])
    
    mem = inp.shape[1]
    
    notes = numpy.zeros(notes_shape)
    time = numpy.zeros(time_shape)

    print("notes, time : ", notes.shape, time.shape)
    # notes[0, :mem, :] = inp #initiating the start

    for tm in range(predict_instances):
        # print("loop", tm)
        y = rnn.predict_b(model_note, inp)
        y_len = rnn.predict_b(model_time, inp_tm)
        # print(y.shape, y_len.shape)
        if 95 < tm < 150 :
            # print(y.shape, y_len.shape)
            pass

        y_len[0] = dutils.arg_max(y_len[0])
        
        for i in range(y.shape[1]):
            y[0, i] = dutils.arg_octave_max(y[0, i])    

        notes[0, tm] = y[0]
        time[0, tm] = y_len[0]

        #Note Value
        inp = shift(inp, axis= 1)
        add_flatroll(inp, y)
        
        #Time Length
        inp_tm = shift(inp_tm, axis=1)
        add_flatroll(inp_tm, y_len)
        
        pass
    return notes, time


def rnn_dense_player(models, ini, ip_memory = None, predict_instances= 400): #works on tick
    #models = (model_n_rnn, model_t_rnn, drum_n_rnn, drum_t_rnn, drum_dense_1, drum_dense_2)
    #ini = (ini_ip, ini_ip_tm, inp_drm_ip, inp_drm_tm) # pinao note, piano time, drum note, drum time
    model_note, model_time, drum_note, drum_time, dense_1, dense_2 = models
    tick = 0
    delt = 32
    ini_ip, ini_t, ini_drm_ip, ini_drm_t = numpy.array([ini[0]]), numpy.array([ini[1]]), numpy.array([ini[2]]), numpy.array([ini[3]])
    muse_op_piano_n = numpy.zeros((1, 4 * predict_instances) + ini_ip.shape[2:])
    muse_op_piano_t = numpy.zeros((1, 4 * predict_instances) + ini_t.shape[2:])
    muse_op_drum_n = numpy.zeros((1, 4 * predict_instances) + ini_drm_ip.shape[2:])
    print(ini_ip.shape, ini_t.shape, ini_drm_ip.shape, ini_drm_t.shape, "-- ini_ip, ini_tm, inp_drm_ip, inp_drm_tm")

    print(muse_op_piano_n.shape, muse_op_piano_t.shape, "-- muse_op_piano_n, muse_op_piano_t")

    TIME = predict_instances * 4
    ticker = [0, 0]
    tm, dtm = 0, 0
    while tick < TIME:
        if ticker[0] >= 0:
            y = rnn.predict_b(model_note, ini_ip)
            y_len = rnn.predict_b(model_time, ini_t)
        
            y_len[0] = dutils.arg_max(y_len[0])
        
            for i in range(y.shape[1]):
                y[0, i] = dutils.arg_octave_max(y[0, i])    

            muse_op_piano_n[0, tm] = y[0]
            muse_op_piano_t[0, tm] = y_len[0]

            #Note Value
            inp = shift(ini_ip, axis= 1)
            add_flatroll(ini_ip, y)
            
            #Time Length
            inp_tm = shift(ini_t, axis=1)
            add_flatroll(ini_t, y_len)

            ticker[0] = dutils.tm(y_len[0])
            tm += 1
        if ticker[1] >=0 :
            #write here code
            y = rnn.predict_b(drum_note, ini_drm_ip)

            for i in range(y.shape[1]):
                y[0, i] = dutils.arg_octave_max(y[0, i])

            muse_op_drum_n[0, dtm] = y[0]

            # print(y.shape, "y . .. ")
            
            y_ = numpy.zeros((3, 2, 16))
            y_[0] = ini_ip[0, -1, 0]
            y_[1] = ini_drm_ip[0, -1, 0]
            y_[2] = ini_drm_ip[0, -1, 1]
            
            # print(y_,y_.shape, "-- y_")

            y_1 = rnn.predict_dense(dense_1, y_)
            y_2 = rnn.predict_dense(dense_2, y_)

            # print(y_1, y_2, "-- y_1, y_2")
            # print(y_1.shape, y_2.shape, "-- y_1, y_2 ---> shape")
            
            y = numpy.zeros((2, 2, 16))

            y[:1] = y_1
            y[1:] = y_2

            #Note Value
            inp_drm_ip = shift(ini_drm_ip, axis= 1)
            add_flatroll(ini_drm_ip, y)
            dtm += 1

            # print(y, y.shape, "y, y.shape .. ")

            # raise NotImplementedError("Err, please analyse one iteration")
            
            ticker[1] = 4



        tick += 1
        ticker[0] -= 1
        ticker[1] -= 1

        if tick % 32 == 0: print("--", end=" ")
    print("\n")
    return muse_op_piano_n, muse_op_piano_t, muse_op_drum_n 


def rnote_player(mnote, ini= None, expected_note= None, TM = 8, ip_memory = 32, DEPTH = 1, predict_instances = 400):
    model_note = rnn.load(mnote) if type(mnote) == str else mnote

    # ip_memory = ip_memory 
    
    inp = numpy.array([ini])
    print('inp note shape : ', inp.shape)
    notes_shape = (1, predict_instances) + inp.shape[2:]
    bs = inp.shape[0]

    predict_instances = (predict_instances // bs) * bs
    
    mem = inp.shape[1]
    
    notes = numpy.zeros(notes_shape)
    time = numpy.zeros((1, predict_instances, 64))
    print(bs, "--bs")
    print("notes, time : ", notes.shape, time.shape)
    # notes[0, :mem, :] = inp #initiating the start

    for tm in range(0, predict_instances):
        # print("loop", tm)
        print('inp : ', inp.shape)
        inp = numpy.reshape(inp, (1, ip_memory,  -1))
        y = rnn.predict_b(model_note, inp)
        y = numpy.reshape(y, (1, DEPTH, 2, 16))
        y_len = numpy.zeros((1, 64))
        y_len[ :, TM] = 1
        # print(y.shape, " --------------")
        
        # for j in range(bs):
        #     y_len[j] = dutils.arg_max(y_len[j])
        for j in range(bs):
            # print('y : ', y, y.shape)
            for i in range(y.shape[1]):

                y[j, i] = dutils.arg_octave_max(y[j, i])    

        notes[0, tm : tm + bs] = y
        time[0, tm : tm + bs] = y_len

        #Note Value
        inp = numpy.reshape(inp, (1, ip_memory, DEPTH, 2, 16))
        inp = shift(inp, axis= 1)
        add_flatroll(inp, y)
        
        #Time Length
        # inp_tm = shift(inp_tm, axis=1)
        # add_flatroll(inp_tm, y_len)
        
        pass
    return enc_deco.octave_to_sFlat(notes), enc_deco.enc_tm_to_tm(time)


def shift(x,  off = 1, axis = 2):
    return numpy.roll(x, -1 * off, axis)

def add_pianoroll(x, y, axis = 2):
    if x.shape[1] != y.shape[1]: raise AttributeError("x[c, : , d] or x.shape[1], and y.shape[0] should be same. ") 
    x[0, :, -1] = y[0]

def add_flatroll(x, y, axis = 2):
    if x.shape[2] != y.shape[1]: raise AttributeError("x[c, d , :] or x.shape[2], and y.shape[1] should be same. ") 
    x[0, -1, :] = y[0]
