

from polymuse import dataset, transformer, enc_deco, dutils

from polymuse import rnn, rnn_player
from polymuse import drawer

from matplotlib import pyplot as plt

# from scipy.interpolate import make_interp_spline, BSpline
import numpy

import gc
import warnings
warnings.filterwarnings("ignore")


"""
Test to understand NoteSequence
"""

f1 = 'F:\\rushikesh\\project\\polymuse-future\\midis\\Believer_-_Imagine_Dragons.mid'
f1 = 'F:\\rushikesh\\project\\polymuse-future\\midis\\Fur_Elise_by_Ludwig_Van_Beethoven.mid'
f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\second_track.mid'
f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\drum_test.mid'
f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\Drummer_Piano.mid'
# f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\dataset.mid'
# f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\test.mid'

ns = dataset.to_array(f)
ns = dataset.to_note_sequence(f)
# print(ns)
ins = dataset.get_instrument(ns, 0)
ar = dataset.ns_to_tarray(ns)
# print(ar, ar.shape, "--tarray")
# s = dataset.tarray_to_sFlat_roll(ar)



# pret = dataset.to_pretty_midi(f)
# fs = 8
# p = dataset.to_pianoroll(pret, fs)
# sf = dataset.pianoroll_to_sFlat_roll(p, 3)
# print(sf, sf.shape, "--roll")
# raise NotImplementedError("Please stop now")



# # ns1 = dataset.to_array(f1)
# ns1 = dataset.to_note_sequence(f1)
# # print(ns)
# ins1 = dataset.get_instrument(ns1, 0)
# ar1 = dataset.ns_to_tarray(ns1)


# print("ar : ", ar.shape)
# sroll = dataset.ns_tarray_to_sFlat(ar, 2)
# print(sroll, ar.shape, sroll.shape)

# s5roll = dataset.ns_tarray_to_sFlat(ar, 5)
# print(s5roll, ar.shape, s5roll.shape)
 
# enc = enc_deco.sFlat_to_octave(sroll)


# x, y = dataset.prepare_sFlat_data(enc, enc_shpae=(2, 16), depth = 2)
# x, y = x[0], y[0]
# print(x, y, x.shape, y.shape)

# dec = enc_deco.octave_to_sFlat(enc)
# print(dec, dec.shape)

"""
Notesequence to and from
"""

# ns_ = dataset.tarray_to_ns(ar, ['guitar'])
# print(ns_)
# print(ns_.notes[0])
# mid = dataset.ns_to_pretty_midi(ns_)
# print(mid)
# print(mid.instruments)
# dataset.ns_to_midi(ns_, 'test.mid')
# mid.write('test.mid')
"""
Testing transformer.first_derivative
"""

#  

"""
Testing ns_tarray_to_sFlat 
""" 
# print(ar, ar.shape)
# sroll = dataset.ns_tarray_to_sFlat(ar, 2)
# print(sroll, ar.shape, sroll.shape)

# s5roll = dataset.ns_tarray_to_sFlat(ar, 5)
# print(s5roll, ar.shape, s5roll.shape)


"""
Testing encoding decoding, prepare data
"""
# enc = enc_deco.sFlat_to_octave(sroll)
# print(enc, enc.shape)

# x, y = dataset.prepare_sFlat_data(enc, enc_shpae=(2, 16), depth = 2)
# print(x, y, x.shape, y.shape)

# dec = enc_deco.octave_to_sFlat(enc)
# print(dec, dec.shape)

"""
Drawer testing
"""

# ft = "F:\\rushikesh\\project\\polymuse-future\\hist\\gTsF_h_512_m_oct_desp_v1___b_64_e_200_d_0.3.json"
# fn = "F:\\rushikesh\\project\\polymuse-future\\hist\\g_h_512_m_oct_desp_v1___b_64_e_200_d_0.3.json"

# drawer.draw_json_loss_acc(fn, ft)

"""

Curve fittling 
"""     

# d = numpy.array([0, 1, 2, 3, 4, 5 ,6 ,7 ,8 ,9 ,10 ,11, 12])
# y = numpy.array([0, 0.3, 0.6, 0.8, 0.8, 0.6, 0.4, 0.25, 0.45, 0.7, 0.7, 0.55, 0.4])

# xnew = numpy.linspace(d.min(),d.max(),300) #300 represents number of points to make between T.min and T.max

# spl = make_interp_spline(d, y, k=3) #BSpline object
# smooth = spl(xnew)

# y_ = -.2932230546  * numpy.sin(xnew + 1.347517745) ** .9919552825 + .5062286427

# y_ = -.2936547276 * numpy.sin(.9735367259 * xnew + 1.377456873) + .5055869885

# y_= 0.04875794598 * numpy.log(41028.86849 *numpy.log(xnew + 1.000024775))

# plt.plot(xnew, smooth, color = 'red', label = 'Original')

# plt.plot(xnew, y_, color = 'blue', label = 'function')

# plt.show()

"""
building sFlat model
"""
# sroll = dataset.ns_tarray_to_sFlat(ar, 2)
# print(sroll, ar.shape, sroll.shape)
 
# enc = enc_deco.sFlat_to_octave(sroll)

# x, y = dataset.prepare_sFlat_data(enc, enc_shpae=(2, 16), depth = 2)
# print("x, y, ", x.shape, y.shape)
# x , y = x[0, : , :, :1], y[0, :, :1]
# print("x, y, ", x.shape, y.shape)

# batch_size = 128
# dropout = 0.3
# cell_count = 512
# epochs = 200
# model_name = 'oactave_v2_s_1_'
# #2_s_1_

# model = 'F:\\rushikesh\\project\\polymuse-future\polymuse\\h5_models\\gsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"

# model = rnn.build_sFlat_model(x, y, model_name, cell_count = cell_count, epochs = epochs, batch_size = batch_size, dropout = dropout)
# # model = rnn.load(model)

# ip = numpy.zeros(x[:1].shape)

# print("Input Shape : ", ip.shape)

# yn = rnn.predict_b(model, ip)
# print("yn shape : ", yn.shape)
# x = numpy.arange(16)

# y00 = yn[0, 0, 0]
# y01 = yn[0, 0, 1]
# # y10 = yn[0, 1, 0]
# # y11 = yn[0, 1, 1]

# y_00 = y[0, 0, 0]
# y_01 = y[0, 0, 1]
# # y_10 = y[0, 1, 0]
# # y_11 = y[0, 1, 1]

# plt.plot(x, y00, color = 'g', label = '00')
# plt.plot(x, y01, color = 'y', label = '01')
# # plt.plot(x, y10, color = 'b', label = '10')
# # plt.plot(x, y11, color = 'r', label = '11')


# plt.plot(x, y_00, color = 'black', label = '-00')
# plt.plot(x, y_01, color = 'cyan', label = '-01')
# # plt.plot(x, y_10, color = 'indigo', label = '-10')
# # plt.plot(x, y_11, color = 'brown', label = '-11')


# plt.legend()

# plt.show()

# print(yn.shape)
# print('built sucessfull')

"""
Time based model
"""

ip_memory = 32

tm = dataset.ns_tarray_to_time(ar[0:1])
print("tm : ", tm.shape)

tm[tm > 63] = 63
print("tm : ", tm.shape)
# print(tm, " --tm ")
tm = dutils.trim_axis2(tm)
tm = enc_deco.tm_to_enc_tm(tm)
print('tm : ', tm.shape)
x, y = dataset.prepare_time_data(tm, enc_shape = (64, ), ip_memory= 32)
print('x, y : ', x.shape, y.shape)
x , y = x[0], y[0]
# print(x, '--x time')
print('x, y : ', x.shape, y.shape)


# li  = dataset.merge_tarray(ar[:1], ar1[:1])

DEPTH = 3
DEPZERO = [0 for _ in range(DEPTH)]

# print(li, "------ li")
print(ar.shape, "--ar1")
# sroll = dataset.ns_tarray_to_sFlat(li[:1], DEPTH)
sroll = dataset.ns_tarray_to_sFlat(ar[:1], DEPTH)
print('sroll : ', sroll.shape)
shape_info = sroll.shape
# cp_t_arr = cp_t_arr[:, :, : tm + 1]
# sroll = sroll[~numpy.all(sroll == DEPZERO, axis=2)]
print("sroll : ", sroll.shape, "--sroll")
 
# sroll = dataset.add_lead_in_sFlat(sroll, lead)

# print("sroll : ", sroll.shape, "--sroll")


sroll = numpy.reshape(sroll, sroll.shape) #For single track
enc = enc_deco.sFlat_to_octave(sroll)
print('enc : ', enc.shape)
x_nt, y_nt = dataset.prepare_sFlat_data(enc, enc_shape=(2, 16), ip_memory=32, depth = DEPTH)
print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)
x_nt, y_nt = x_nt[0, : , :, :1], y_nt[0, :, :1]
print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)

# print("x, y : ", x.shape, y.shape)

batch_size = 32
dropout = 0.3
cell_count = 512
epochs = 200

model_name = 'M_custom_loss_track_s1_v11_'
#2_s_1_ gsF_512_m_oct_desp_v1___b_64_e_200_d_0.3


# m_note = 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\gsF_512_m_oct_desp_v1___b_64_e_200_d_0.3.h5'
# m_time = 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\gTsF_512_m_oct_desp_v1___b_64_e_200_d_0.3.h5'
# model = 'F:\\rushikesh\\project\\polymuse-future\polymuse\\h5_models\\gTsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"

# m_note = rnn.build_sFlat_model(x_nt, y_nt, model_name, cell_count = cell_count, epochs = epochs, batch_size = batch_size, dropout = dropout)
# m_time = rnn.build_time_sFlat_model(x, y, model_name, cell_count = cell_count, epochs = epochs, batch_size = batch_size, dropout = dropout)

ft = 'F:\\rushikesh\\project\\polymuse-future\\hist\\gTsF_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
fn = 'F:\\rushikesh\\project\\polymuse-future\\hist\\g_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
    
# drawer.draw_json_loss_acc(fn, ft)

m_note = 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\gsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"
m_time ='F:\\rushikesh\\project\\polymuse-future\\h5_models\\gTsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"


m_note = rnn.load(m_note)
m_time = rnn.load(m_time)


ip_nt = numpy.array(x_nt[34: 66])
ip = numpy.array(x[34: 66])

print('ip_nt, ip : ', ip_nt.shape, ip.shape)

print('x, y : ', x.shape, y.shape)
print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)

note, time = rnn_player.rsingle_note_time_play(m_note, m_time, ip_nt, ip, y_nt, y, ip_memory, 450)

print("note, time : ", note.shape, time.shape)

# for i in range(10):
#     print("time : ", time[0, i])

note, time = enc_deco.octave_to_sFlat(note), enc_deco.enc_tm_to_tm(time)


# for i in range(10):
#     print("time : ", time[0, i])

print("ENC :: note, time : ", note.shape, time.shape)


t_array = dataset.snote_time_to_tarray(note, time)
print("t_array : ", t_array)
ns_ = dataset.tarray_to_ns(t_arr= t_array, instruments= ['harmonica'])

m_path = 'F:\\rushikesh\\project\\polymuse-future\\midis\\' + model_name + " _NT_" + "gsF_512_m_oactave_v2_s_1___b_128_e_200_d_0.3__" + "gTsF_512_m_oactave_tm_v1___b_64_e_200_d_0.3.mid"

dataset.ns_to_midi(ns_, m_path)




# print("Input Shape : ", ip.shape)
# y_00, y00 = rnn.evalulate(model, x, y)
# yn = rnn.predict_b(model, ip)
# print("yn shape : ", yn.shape)
# y_00, y00 = y_00[:200], y00[:200]
# x = numpy.arange(len(y00))

# # y00 = yn[0]
# # # y01 = yn[0, 0, 1]
# # # y10 = yn[0, 1, 0]
# # # y11 = yn[0, 1, 1]

# # y_00 = y[34]
# # # y_01 = y[0, 0, 1]
# # # y_10 = y[0, 1, 0]
# # # y_11 = y[0, 1, 1]

# plt.plot(x, y00, color = 'g', label = '00')
# # # plt.plot(x, y01, color = 'y', label = '01')
# # # plt.plot(x, y10, color = 'b', label = '10')
# # # plt.plot(x, y11, color = 'r', label = '11')


# plt.plot(x, y_00, color = 'y', linestyle  = 'dashed', label = '-00')
# # # plt.plot(x, y_01, color = 'cyan', label = '-01')
# # # plt.plot(x, y_10, color = 'indigo', label = '-10')
# # # plt.plot(x, y_11, color = 'brown', label = '-11')


# plt.legend()

# plt.show()


"""
Roll Model

"""



# ip_memory = 64

# # rsf = numpy.reshape(sf, (sf.shape[1], sf.shape[0], sf.shape[2]))


# # print(rsf, rsf.shape, " --rsf")

# DEPTH = 3
# DEPZERO = [0 for _ in range(DEPTH)]

# # print(li, "------ li")

# # sroll = dataset.ns_tarray_to_sFlat(li[:1], DEPTH)
# # lead = dataset.ns_tarray_to_sFlat(ar1[:1], DEPTH)
# # print('sroll : ', sroll.shape)
# # shape_info = sroll.shape
# # cp_t_arr = cp_t_arr[:, :, : tm + 1]
# # sroll = sroll[~numpy.all(sroll == DEPZERO, axis=2)]
# # print("sroll : ", sroll.shape, "--sroll")
 
# # sroll = dataset.add_lead_in_sFlat(sroll, lead)

# # print("sroll : ", sroll.shape, "--sroll")


# # sroll = numpy.reshape(sf, sf.shape) #For single track
# enc = enc_deco.sFlat_to_octave(sf)
# gc.collect()
# print('enc : ', enc.shape)
# x_nt, y_nt = dataset.prepare_sFlat_roll_data(enc, enc_shape=(2, 16), ip_memory=ip_memory, depth = DEPTH)
# print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)
# x_nt, y_nt = x_nt[:512], y_nt[:512]
# print(x_nt, y_nt, x_nt.shape, y_nt.shape)

# # print("x, y : ", x.shape, y.shape)

# ## v1 = 25, v2 = 20, v3 = 73 epochs  --- ip_meomry = 32

# batch_size = 512
# dropout = 0.3
# cell_count = 512
# epochs = 100

# model_name = 'oct_trk_roll_s3_v3_'
# #2_s_1_ gsF_512_m_oct_desp_v1___b_64_e_200_d_0.3

# # m_note = 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\gsF_512_m_oct_desp_v1___b_64_e_200_d_0.3.h5'
# # m_time = 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\gTsF_512_m_oct_desp_v1___b_64_e_200_d_0.3.h5'
# # model = 'F:\\rushikesh\\project\\polymuse-future\polymuse\\h5_models\\gTsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"

# # m_time = rnn.build_time_sFlat_model(x, y, model_name, cell_count = cell_count, epochs = epochs, batch_size = batch_size, dropout = dropout)
# m_note = rnn.build_sFlat_model(x_nt, y_nt, model_name, cell_count = cell_count, epochs = epochs, batch_size = batch_size, dropout = dropout)

# # ft = 'F:\\rushikesh\\project\\polymuse-future\\hist\\gTsF_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
# fn = 'F:\\rushikesh\\project\\polymuse-future\\hist\\g_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
    
# drawer.draw_json_loss_acc(fn, ft)

# # m_note = 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\gsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"
# # m_time ='F:\\rushikesh\\project\\polymuse-future\\h5_models\\gTsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"


# # m_note = rnn.load(m_note)
# # m_time = rnn.load(m_time)


# ip_nt = numpy.zeros(x_nt[34].shape)
# ip = numpy.zeros(x[34].shape)

# print('ip_nt, ip : ', ip_nt.shape, ip.shape)

# print('x, y : ', x.shape, y.shape)
# print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)

# note, time = rnn_player.rsingle_note_time_play(m_note, m_time, ip_nt, ip, y_nt, y, ip_memory, 150)

# print("note, time : ", note.shape, time.shape)

# # for i in range(10):
# #     print("time : ", time[0, i])

# note, time = enc_deco.octave_to_sFlat(note), enc_deco.enc_tm_to_tm(time)


# # for i in range(10):
# #     print("time : ", time[0, i])

# print("ENC :: note, time : ", note.shape, time.shape)


# t_array = dataset.snote_time_to_tarray(note, time)
# print("t_array : ", t_array)
# ns_ = dataset.tarray_to_ns(t_arr= t_array, instruments= ['guitar'])

# m_path = 'F:\\rushikesh\\project\\polymuse-future\\midis\\' + model_name + " _NT_" + "gsF_512_m_oactave_v2_s_1___b_128_e_200_d_0.3__" + "gTsF_512_m_oactave_tm_v1___b_64_e_200_d_0.3.mid"

# dataset.ns_to_midi(ns_, m_path)

"""
drum test
"""

# print(ar)