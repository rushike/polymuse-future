
from polymuse import rnn, dutils, dataset, dataset2 as d2, enc_deco, rnn_gpu
from polymuse.losses import rmsecat
import tensorflow as tf
import numpy, random
"""
rnn_player -- capable of playing/generating the music output as octave/time encoded representation

These also includes two most important functions :
    * shift:
    * add_flatroll: 
    * sample
"""

def polymuse_player(model, ini, expected_note= None, TM = 8, ip_memory = 32, DEPTH = 1, predict_instances = 400, bits= 8):
    # print("ini len : ", len(ini))
    # print("ini len 2 : ", [ini[i].shape  for i  in range(len(ini))])
    # print("inin shape 3 : ", [list(ini[i][j].shape for j in range(len(ini[i])) ) for i  in range(len(ini))])
    #ini shape = (type, batch_tm, ip_memory, depth)
    lead_ini, chorus_ini, drum_ini = numpy.array([ini[0][8]]), numpy.array([ini[1][8]]), numpy.array([ini[2][8]])

    init = [lead_ini, chorus_ini, drum_ini]

    #init shape = (type, 1, ip_memory, depth * bits)

    print("INPUT SHAPE lead, chorus, drum : ", lead_ini.shape, chorus_ini.shape, drum_ini.shape)

    lead_notes_shape = (1, predict_instances, 1) 
    chorus_notes_shape = (1, predict_instances, 3) 
    drum_notes_shape = (1, predict_instances, 2) 

    lead_notes = numpy.zeros(lead_notes_shape)
    chorus_notes = numpy.zeros(chorus_notes_shape)
    drum_notes = numpy.zeros(drum_notes_shape)

    notes = [lead_notes, chorus_notes, drum_notes]
    
    for tm in range(predict_instances):
        pred = rnn_gpu.predict_w(model, [lead_ini, chorus_ini, drum_ini])
        
        for j in range(len(pred)): # each model iterator
            y = numpy.reshape(pred[j][0], (pred[j][0].shape[0] // 32, 2, 16))
            b = numpy.zeros((1, 8 * y.shape[0]))
            for i in range(y.shape[0]): # depth iterator
                ocn, freqn = numpy.argmax(y[i, 0]), dutils.sample(y[i, 1], temperature= 1.1375)
                b[0, i * 8 : (i + 1) * 8] = enc_deco.binarr(ocn * 12 + freqn, bits= 8) # setting notes to repective track
                notes[j][0, tm, i] = ocn * 12 + freqn
                pass
            
            init[j] = shift(init[j], axis= 1)
            # print("The Shape : ", init[j].shape, b.shape, y.shape, pred[j].shape)
            add_flatroll(init[j], b)
            pass

        pass


    pass
    lead_sf = d2.sFlatroll_to_ns_tarray(notes[0], 2)
    chorus_sf = d2.sFlatroll_to_ns_tarray(notes[1], 2)
    drum_sf = d2.sFlatroll_to_ns_tarray(notes[2], 2, tm = 4)
    tarray = numpy.zeros((3, max([lead_sf.shape[1], chorus_sf.shape[1], drum_sf.shape[1]]), 4))
    print("lead, chorus, drum : ", lead_sf.shape, chorus_sf.shape, drum_sf.shape)

    tarray[0, :lead_sf.shape[1]] = lead_sf
    tarray[1, :chorus_sf.shape[1]] = chorus_sf
    tarray[2, :drum_sf.shape[1]] = drum_sf
    print("tarray : ", tarray.shape)
    # tarr = numpy.zeros()
    # print(notes, "--notes")
    return tarray




def shift(x,  off = 1, axis = 2):
    return numpy.roll(x, -1 * off, axis)

def add_pianoroll(x, y, axis = 2):
    if x.shape[1] != y.shape[1]: raise AttributeError("x[c, : , d] or x.shape[1], and y.shape[0] should be same. ") 
    x[0, :, -1] = y[0]

def add_flatroll(x, y, axis = 2):
    if x.shape[2] != y.shape[1]: raise AttributeError("x[c, d , :] or x.shape[2], and y.shape[1] should be same. ") 
    x[0, -1, :] = y[0]

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)