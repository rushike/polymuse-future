import os, numpy, random

from polymuse import dataset, transformer, enc_deco, dutils, dataset2 as d2, pattern

from keras.models import load_model  

from polymuse import rnn_player, rnn

def play_multitrack(lead, chorus, drum, ini = None, predict_instances = 50, ip_memory = 32, raw = False):
    ip_memory = 32
    print("Lead :", lead[0].input_shape, lead[1].input_shape)
    
    ip_pat = random.choice([pattern.ip_patterns])
    en = d2.ip_patterns_to_octave(ip_pat)
    ip_nt, ip = en[0, :32] , numpy.zeros((ip_memory, 64)) #numpy.zeros((ip_memory, 1, 2, 16))
    #lead
    note, time = rnn_player.rsing_note_time_play(lead[0], lead[1], ip_nt, ip, ip_memory, predict_instances= predict_instances)
    note, time = enc_deco.octave_to_sFlat(note), enc_deco.enc_tm_to_tm(time)
    t_array1 = dataset.snote_time_to_tarray(note, time)
    print(t_array1.shape, " tarray1")

    ip_nt, ip = numpy.zeros((ip_memory, 3, 2, 16)), numpy.zeros((ip_memory, 64))
    #chorous
    note, time = rnn_player.rsingle_note_time_play(chorus[0], chorus[1], ip_nt, ip, ip_memory, predict_instances=predict_instances)
    note, time = enc_deco.octave_to_sFlat(note), enc_deco.enc_tm_to_tm(time)
    t_array2 = dataset.snote_time_to_tarray(note, time)
    print(t_array2.shape, " tarray2")

    ip_nt, ip = numpy.zeros((ip_memory, 2, 2, 16)), numpy.zeros((ip_memory, 64))
    #drum
    note, time = rnn_player.rsingle_note_time_play(drum[0], drum[1], ip_nt, ip, ip_memory, predict_instances=predict_instances)
    note, time = enc_deco.octave_to_sFlat(note), enc_deco.enc_tm_to_tm(time)
    t_array3 = dataset.snote_time_to_tarray(note, time)
    print(t_array3.shape, " tarray3")
    
    if raw: return t_array1, t_array2, t_array3

    mx_tm = max([t_array1.shape[1], t_array2.shape[1], t_array3.shape[1]])


    t_array = numpy.zeros((3, mx_tm, 4))

    t_array[0, :t_array1.shape[1]] = t_array1[0]
    t_array[1,  :t_array2.shape[1]] = t_array2[0]
    t_array[2,  :t_array3.shape[1]] = t_array3[0]

    return t_array

def dual_pianodrum_rollsto_tarray(mn, mt, md):
    note, time = enc_deco.octave_to_sFlat(mn), enc_deco.enc_tm_to_tm(mt)
    t_array1 = dataset.snote_time_to_tarray(note, time)
    print(t_array1.shape, " tarray1")

    note = enc_deco.octave_to_sFlat(md)
    t_array2 = dataset.snote_time_to_tarray(note, None, deltam=4)
    print(t_array2.shape, " tarray2")
   
    mx_tm = max([t_array1.shape[1], t_array2.shape[1]])

    t_array = numpy.zeros((2, mx_tm, 4))

    t_array[0, :t_array1.shape[1]] = t_array1[0]
    t_array[1,  :t_array2.shape[1]] = t_array2[0]

    return t_array


def get_3_multitrack():
    home = 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\'
    dirs= ['lead', 'chorus', 'drum']
    op_models = []
    for p in dirs:
        models = os.listdir(home + p)
        if models[0].startswith("gsF"):
            mv = load_model(home + p + "\\" + models[0])
            mt = load_model(home + p + "\\" + models[1])
        elif models[1].startswith("gTsF"):
            mt = load_model(home + p + "\\" + models[0])
            mv = load_model(home + p + "\\" + models[1])
        op_models.append([mv, mt])
        # models = [home + p + "\\" + m for m in models]
        # print(models)
    return op_models



def sync_play_multitrack(lead, chorus, drum, ini, predict_instances = 500, ip_memory = 32):
    tarr1, tarr2, tarr3 = play_multitrack(lead, chorus, drum, ini, predict_instances, ip_memory, raw=True)
    
    print(tarr1.shape, tarr2.shape, tarr3.shape, "---- tarr1, tarr2, tarr3")

    sflat = d2.ns_tarray_to_sFlatroll(tarr1)

    print(sflat, sflat.shape, " --- sflatroll ")

    tarr2, tarr3 = d2.sync_tarray(t_arr1=tarr2, t_arr2=tarr3)
    # print(tarr1.shape, tarr2.shape, tarr3.shape, "-----Shape tarr 0, 1, 2")

    tarr1, tarr2 = d2.sync_tarray(t_arr1=tarr1, t_arr2=tarr2)

    # print(tarr1,tarr2, tarr3, "--tarr 0, 1, 2 ")
    # print(tarr1.shape, tarr2.shape, tarr3.shape, "-----Shape tarr 0, 1, 2")

    tarr1, tarr3 = d2.sync_tarray(tarr1, tarr3)

    # print(tarr1.shape, tarr2.shape, tarr3.shape, "-----Shape tarr 0, 1, 2")

    
    t_array = numpy.zeros((3, len(tarr1[0]), 4))

    t_array[0, : tarr1.shape[1]] = tarr1[0]
    t_array[1,  :tarr2.shape[1]] = tarr2[0]
    t_array[2,  :tarr3.shape[1]] = tarr3[0]

    return t_array




def sync_play_multitrack2(lead, chorus, drum, ini, predict_instances = 500, ip_memory = 32):
    #models = (model_n_rnn, model_t_rnn, drum_n_rnn, drum_t_rnn, drum_dense_1, drum_dense_2)
    #ini = (ini_ip, ini_ip_tm, inp_drm_ip, inp_drm_tm) # pinao note, piano time, drum note, drum time
    model_note, model_time, ch_note, ch_time, drum_note, drum_time = lead[0], lead[1], chorus[0], chorus[1], drum[0], drum[1]
    
    tick = 0
    delt = 32

    ini_ip, ini_t, ini_ch, ini_ch_t, ini_drm_ip = numpy.array([ini[0]]), numpy.array([ini[1]]), numpy.array([ini[2]]), numpy.array([ini[3]]), numpy.array([ini[4]])
    
    muse_op_piano_n = numpy.zeros((1, 4 * predict_instances) + ini_ip.shape[2:])
    muse_op_piano_t = numpy.zeros((1, 4 * predict_instances) + ini_t.shape[2:])
    
    muse_op_ch = numpy.zeros((1, 4 * predict_instances) + ini_ch.shape[2:])
    muse_op_ch_t = numpy.zeros((1, 4 * predict_instances) + ini_ch_t.shape[2:])
    
    muse_op_drum_n = numpy.zeros((1, 4 * predict_instances) + ini_drm_ip.shape[2:])
    # print(ini_ip.shape, ini_t.shape, ini_drm_ip.shape, ini_drm_t.shape, "-- ini_ip, ini_tm, inp_drm_ip, inp_drm_tm")

    print(muse_op_piano_n.shape, muse_op_piano_t.shape, "-- muse_op_piano_n, muse_op_piano_t")
    print(muse_op_ch.shape, muse_op_ch_t.shape, "-- muse_op_ch_n, muse_op_ch_t")
    print(muse_op_drum_n.shape, "-- muse_op_drum_n")

    TIME = predict_instances * 4
    ticker = [0, 0, 0]
    tm, tmch, dtm = 0, 0, 0
    while tick < TIME:
        if ticker[0] <= 0:
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

        if ticker[1] <= 0:
            y = rnn.predict_b(ch_note, ini_ch)
            y_len = rnn.predict_b(ch_time, ini_ch_t)
        
            y_len[0] = dutils.arg_max(y_len[0])
        
            for i in range(y.shape[1]):
                y[0, i] = dutils.arg_octave_max(y[0, i])    

            muse_op_ch[0, tmch] = y[0]
            muse_op_ch_t[0, tmch] = y_len[0]

            #Note Value
            inp = shift(ini_ch, axis= 1)
            add_flatroll(ini_ch, y)
            
            #Time Length
            inp_tm = shift(ini_ch_t, axis=1)
            add_flatroll(ini_ch_t, y_len)

            ticker[1] = dutils.tm(y_len[0])
            tmch += 1

        if ticker[2] <= 0 :
            #write here code
            y = rnn.predict_b(drum_note, ini_drm_ip)

            for i in range(y.shape[1]):
                y[0, i] = dutils.arg_octave_max(y[0, i])

            muse_op_drum_n[0, dtm] = y[0]

            # print(y.shape, "y . .. ")

            # print(y_1, y_2, "-- y_1, y_2")
            # print(y_1.shape, y_2.shape, "-- y_1, y_2 ---> shape")


            #Note Value
            inp_drm_ip = shift(ini_drm_ip, axis= 1)
            add_flatroll(ini_drm_ip, y)
            dtm += 1

            # print(y, y.shape, "y, y.shape .. ")

            # raise NotImplementedError("Err, please analyse one iteration")
            
            ticker[2] = 4

        tick += 1
        ticker[0] -= 1
        ticker[1] -= 1
        ticker[2] -= 1

        if tick % 32 == 0: print("--", end=" ")
    print("\n")

    muse_op_piano_n, muse_op_piano_t = enc_deco.octave_to_sFlat(muse_op_piano_n), enc_deco.enc_tm_to_tm(muse_op_piano_t)
    t_array1 = dataset.snote_time_to_tarray(muse_op_piano_n, muse_op_piano_t)

    muse_op_ch, muse_op_ch_t = enc_deco.octave_to_sFlat(muse_op_ch), enc_deco.enc_tm_to_tm(muse_op_ch_t)
    t_array2 = dataset.snote_time_to_tarray(muse_op_ch, muse_op_ch_t)

    muse_op_drum_n = enc_deco.octave_to_sFlat(muse_op_drum_n)
    t_array3 = dataset.snote_time_to_tarray(muse_op_drum_n, None, deltam=4)

    mx_tm = max([t_array1.shape[1], t_array2.shape[1], t_array3.shape[1]])

    t_array = numpy.zeros((3, mx_tm, 4))

    t_array[0, :t_array1.shape[1]] = t_array1[0]
    t_array[1,  :t_array2.shape[1]] = t_array2[0]
    t_array[2,  :t_array3.shape[1]] = t_array3[0]


    return t_array#muse_op_piano_n, muse_op_piano_t, muse_op_ch, muse_op_ch_t, muse_op_drum_n 


def shift(x,  off = 1, axis = 2):
    return numpy.roll(x, -1 * off, axis)

def add_pianoroll(x, y, axis = 2):
    if x.shape[1] != y.shape[1]: raise AttributeError("x[c, : , d] or x.shape[1], and y.shape[0] should be same. ") 
    x[0, :, -1] = y[0]

def add_flatroll(x, y, axis = 2):
    if x.shape[2] != y.shape[1]: raise AttributeError("x[c, d , :] or x.shape[2], and y.shape[1] should be same. ") 
    x[0, -1, :] = y[0]


