from polymuse import dataset, transformer, enc_deco, dutils, dataset2 as d2
from polymuse import multi_track


from polymuse import rnn, rnn_player
from polymuse import drawer

import pyfiglet

import os

from matplotlib import pyplot as plt

# from scipy.interpolate import make_interp_spline, BSpline
import numpy
big = pyfiglet.Figlet(font="slant")

HOME = os.getcwd() 

def build_piano_stateful_model(dataset_path, lr = 0.01, model_name = "note", cell_count = 512, batch_size = 32, dense_count = 96, epochs = 300, dropout = 0.3):
    ns = dataset.to_note_sequence(dataset_path)
    
    ar = dataset.ns_to_tarray(ns, resolution=64)
    
    ip_memory = 32

    print(big.renderText("NOTE STATE"))
    
    tm = dataset.ns_tarray_to_time(ar[:1])
    print("tm : ", tm.shape)
    tm[tm > 31] = 31
    tm = dutils.trim_axis2(tm)
    
    tm = enc_deco.tm_to_enc_tm(tm)
    print('tm : ', tm.shape)
    
    x_t, y_t = dataset.prepare_time_data(tm, enc_shape = (64, ), ip_memory= 32)
    print('x, y : ', x_t.shape, y_t.shape)
    x_t , y_t = x_t[0], y_t[0]
    print('xn_t, y : ', x_t.shape, y_t.shape)


    DEPTH = 1

    sroll_note = dataset.ns_tarray_to_sFlat(ar[:1], DEPTH)

    print("sroll_note : ", sroll_note.shape, " --- sroll_note")

    enc = enc_deco.sFlat_to_octave(sroll_note)
    print('enc : ', enc.shape)
    x_nt, y_nt = dataset.prepare_sFlat_data(enc, enc_shape=(2, 16), ip_memory=32, depth = DEPTH)

    print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)
    x_nt, y_nt = x_nt[0, : , :, :1], y_nt[0, :, :1]
    print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)

    m_note = rnn.build_sFlat_stateful_model(x_nt, y_nt, model_name, cell_count = cell_count, epochs = epochs, batch_size = batch_size, dropout = dropout)
    m_time = rnn.build_time_sFlat_stateful_model(x_t, y_t, model_name, cell_count = cell_count, epochs = epochs, batch_size = batch_size, dropout = dropout)

    ft = 'F:\\rushikesh\\project\\polymuse-future\\hist\\piano\\stateful\\gTsF_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
    fn = 'F:\\rushikesh\\project\\polymuse-future\\hist\\piano\\stateful\\g_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
        
    drawer.draw_json_loss_acc(fn, ft)

    return m_note, m_time, (x_nt[:batch_size].shape, x_t[: batch_size ].shape), (x_nt, x_t)


def build_piano_model(dataset_path, lr = 0.01, model_name = "note", cell_count = 512, batch_size = 32, dense_count = 96, epochs = 300, dropout = 0.3):
    ns = dataset.to_note_sequence(dataset_path)
    
    ar = dataset.ns_to_tarray(ns, resolution=64)
    print(ar)
    ip_memory = 32

    print(big.renderText("NOTE STATELESS"))
    
    tm = dataset.ns_tarray_to_time(ar[:1])
    print("tm : ", tm.shape)
    tm[tm > 31] = 31
    tm = dutils.trim_axis2(tm)
    
    tm = enc_deco.tm_to_enc_tm(tm)
    print('tm : ', tm.shape)
    
    x_t, y_t = dataset.prepare_time_data(tm, enc_shape = (64, ), ip_memory= 32)
    print('x, y : ', x_t.shape, y_t.shape)
    x_t , y_t = x_t[0], y_t[0]
    print('xn_t, y : ', x_t.shape, y_t.shape)


    DEPTH = 1

    sroll_note = dataset.ns_tarray_to_sFlat(ar[:1], DEPTH)

    print("sroll_note : ", sroll_note.shape, " --- sroll_note")

    enc = enc_deco.sFlat_to_octave(sroll_note)
    print('enc : ', enc.shape)
    x_nt, y_nt = dataset.prepare_sFlat_data(enc, enc_shape=(2, 16), ip_memory=32, depth = DEPTH)

    print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)
    x_nt, y_nt = x_nt[0, : , :, :1], y_nt[0, :, :1]
    print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)

    m_note = rnn.build_sFlat_model(x_nt, y_nt, model_name, cell_count = cell_count, epochs = epochs, batch_size = batch_size, dropout = dropout)
    m_time = rnn.build_time_sFlat_model(x_t, y_t, model_name, cell_count = cell_count, epochs = epochs, batch_size = batch_size, dropout = dropout)

    ft = HOME + '/hist/piano/stateless/gTsF_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
    fn = HOME + '/hist/piano/stateless/g_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
        
    drawer.draw_json_loss_acc(fn, ft)

    return m_note, m_time, (x_nt[1].shape, x_t[1].shape)

