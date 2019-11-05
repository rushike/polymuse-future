from polymuse import dataset, transformer, enc_deco, dutils, dataset2 as d2, evaluation, constant, data_generator, rnn_h, rnn_gpu, pattern
# from polymuse import multi_track


from polymuse import train, player, loader
from polymuse import rnn_player
# from polymuse import drawer

from matplotlib import pyplot as plt

# from polymuse import builder, player
# from polymuse import pattern

# from scipy.interpolate import make_interp_spline, BSpline
import numpy, random, pprint, copy

import gc, os
import warnings
warnings.filterwarnings("ignore")


"""
Test to understand NoteSequence
"""

# f1 = 'F:\\rushikesh\\project\\polymuse-future\\midis\\Believer_-_Imagine_Dragons.mid'
# f1 = 'F:\\rushikesh\\project\\polymuse-future\\midis\\Fur_Elise_by_Ludwig_Van_Beethoven.mid'
# f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\second_track.mid'
# f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\drum_test.mid'
# f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\Drummer_Piano.mid'
# f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\drum_sync.mid'
# f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\dataset.mid'
# f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\test.mid'

# ns = dataset.to_array(f)
# ns = dataset.to_note_sequence(f)
# # # print(ns)
# # ins = dataset.get_instrument(ns, 0)
# ar = dataset.ns_to_tarray(ns, resolution=64)
# print(ar.shape, '--ar')

# sflatroll = d2.ns_tarray_to_sFlatroll(ar)

# tarr = d2.sFlatroll_to_ns_tarray(sflatroll)


# print(sflatroll, sflatroll.shape, '--- sflatroll .. ')

# print("ar : ", ar.shape, "--tarray")
# s = dataset.tarray_to_sFlat_roll(ar)

"""


"""
# p, y = d2.sync_tarray(ar[:1], ar[1:2])

# print(p, y, "-- p, y")
# for i in range(p.shape[0]):
#     print(p[i], y[i])
"""
drum test
"""

# print(ar , "--tarray")

# rl = d2.tarray_to_sFlat_roll(ar, 3)

# flat = dataset.ns_tarray_to_sFlat(ar, 3)

# print(flat, 'roll')


"""
model loader -- multitrack
"""

# mo = multi_track.get_3_multitrack()
# t_array = multi_track.play_multitrack(mo[0], mo[1], mo[2], predict_instances=400)
# print(t_array, t_array.shape, "--t_array")

# ns_ = dataset.tarray_to_ns(t_arr= t_array, instruments= ['piano', 'guitar','choir aahs'], drm = 2)
# # print(ns_)
# ct, st = 0, -1
# for n in ns_.notes:
#     if st != n.instrument:
#         st = n.instrument
#         ct +=1

# print("st, ct", st, ct)
# m_path = "F:\\rushikesh\\project\\polymuse-future\\midis\\multi\\" + "secomd_multitrack_v2.mid"

# dataset.ns_to_midi(ns_, m_path)


"""
Sync multi track
"""
# mo = multi_track.get_3_multitrack()
# # multi_track.sync_play_multitrack(mo[0], mo[1], mo[2], ini = None)
# # ip_pat = random.choice([pattern.ip_patterns])

# # trr = d2.ip_patterns_to_tarray(ip_pat)
# # print(trr, "--trr")
# # print("trr >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> : ", trr.shape)

# # fal = d2.ip_patterns_to_sFlat(ip_pat, DEPTH = 1)
# # print("flat >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> : ", fal.shape)

# # en = d2.ip_patterns_to_octave(ip_pat)
# # print("enc >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> : ", en.shape)
# # print(random.choice(pattern.ip_patterns))
# ini = ( numpy.zeros((32, 1, 2, 16)), numpy.zeros((32, 64)), numpy.zeros((32, 3, 2, 16)), numpy.zeros((32, 64)), numpy.zeros((32, 2, 2, 16)))
# t_array = multi_track.sync_play_multitrack(mo[0], mo[1], mo[2], ini=ini, predict_instances=400)
# print(t_array, t_array.shape, "--t_array")

# ns_ = dataset.tarray_to_ns(t_arr= t_array, instruments= ['piano', 'guitar','choir aahs'], drm = 2)
# # print(ns_)
# ct, st = 0, -1
# for n in ns_.notes:
#     if st != n.instrument:
#         st = n.instrument
#         ct +=1

# print("st, ct", st, ct)
# m_path = "F:\\rushikesh\\project\\polymuse-future\\midis\\multi\\" + "sync_pattern_multitrack_v2.mid"

# dataset.ns_to_midi(ns_, m_path)

# p_score = evaluation.polyphony_score(t_array)

# print(p_score, p_score.shape, "--p score")

# x = numpy.arange(p_score.shape[0])


# plt.plot(x, p_score)

# plt.show()

# print("Average Polyphony Score : ", sum(p_score) / x.shape[0])

"""
Evalution Output: 
Polyphonic Index : 0 - 1
Tonal Span : > 1, max means better
Scale Consistency : 0 - 1
"""
# 33
"""
drum model
"""

# ip_memory = 32 #input memory : prev notes

# tm = dataset.ns_tarray_to_time(ar[0:1]) #(tracks , time_instance)
# print("tm : ", tm.shape)

# tm[tm > 31] = 31

# # print(tm, " --tm ")
# tm = dutils.trim_axis2(tm)
# tm = enc_deco.tm_to_enc_tm(tm) #(tracks, time_instance, 64)
# print('tm : ', tm.shape)
# x, y = dataset.prepare_time_data(tm, enc_shape = (64, ), ip_memory= 32) #(#tracks ,time_instaances, 32, 64)
# print('x, y : ', x.shape, y.shape)
# x , y = x[0], y[0] #Train one track at time
# # print(x, '--x time')
# print('x, y : ', x.shape, y.shape)


# # li  = dataset.merge_tarray(ar[:1], ar1[:1])

# DEPTH = 2
# DEPZERO = [0 for _ in range(DEPTH)]

# # print(li, "------ li")
# print(ar.shape, "--ar1")
# # sroll = dataset.ns_tarray_to_sFlat(li[:1], DEPTH)
# sroll = dataset.ns_tarray_to_sFlat(ar, DEPTH) #(track, note_instaces, depth)
# # sroll = dataset.ns_tarray_to_sFlat(ar1[:1], DEPTH)
# # sroll = d2.sFlat_to_fFlat(sroll)

# print('sroll : ', sroll.shape)
# shape_info = sroll.shape
# # cp_t_arr = cp_t_arr[:, :, : tm + 1]
# # sroll = sroll[~numpy.all(sroll == DEPZERO, axis=2)]
# print("sroll : ", sroll.shape, "--sroll")
 
# # sroll = dataset.add_lead_in_sFlat(sroll, lead)

# # print("sroll : ", sroll.shape, "--sroll")


# sroll = numpy.reshape(sroll, sroll.shape) #For single track
# enc = enc_deco.sFlat_to_octave(sroll) #(track, note_instances, depth, 2, 16)
# print('enc : ', enc.shape)
# x_nt, y_nt = dataset.prepare_sFlat_data(enc, enc_shape=(2, 16), ip_memory=32, depth = DEPTH) #(track, note_instances, 32, depth, 2, 16)
# print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)
# x_nt, y_nt = x_nt[0, : , :, :2], y_nt[0, :, :2]
# print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)

# # print("x, y : ", x.shape, y.shape)

# batch_size = 32
# dropout = 0.3
# cell_count = 512
# epochs = 100

# model_name = 'oct_drum_adam_track_s2_v9_'
# #2_s_1_ gsF_512_m_oct_desp_v1___b_64_e_200_d_0.3


# # m_note = 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\gsF_512_m_oct_desp_v1___b_64_e_200_d_0.3.h5'
# # m_time = 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\gTsF_512_m_oct_desp_v1___b_64_e_200_d_0.3.h5'
# # model = 'F:\\rushikesh\\project\\polymuse-future\polymuse\\h5_models\\gTsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"

# m_note = rnn.build_sFlat_model(x_nt, y_nt, model_name, cell_count = cell_count, epochs = epochs, batch_size = batch_size, dropout = dropout)
# m_time = rnn.build_time_sFlat_model(x, y, model_name, cell_count = cell_count, epochs = epochs, batch_size = batch_size, dropout = dropout)

# ft = 'F:\\rushikesh\\project\\polymuse-future\\hist\\gTsF_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
# fn = 'F:\\rushikesh\\project\\polymuse-future\\hist\\g_h_' + str(cell_count)+ '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout) + ".json"
    
# drawer.draw_json_loss_acc(fn, ft)

# m_note = 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\gsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"
# m_time ='F:\\rushikesh\\project\\polymuse-future\\h5_models\\gTsF_' +  str(cell_count) + '_m_' + model_name +'__b_' + str(batch_size) + "_e_"+str(epochs) + "_d_" + str(dropout)  + ".h5"


# # m_note = rnn.load(m_note)
# # m_time = rnn.load(m_time)


# ip_nt = numpy.zeros(x_nt[34].shape)
# ip = numpy.zeros(x[34].shape)

# print('ip_nt, ip : ', ip_nt.shape, ip.shape)

# print('x, y : ', x.shape, y.shape)
# print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)

# note, time = rnn_player.rsingle_note_time_play(m_note, m_time, ip_nt, ip, y_nt, y, ip_memory, 350)

# print("note, time : ", note.shape, time.shape)

# note, time = enc_deco.octave_to_sFlat(note), enc_deco.enc_tm_to_tm(time)

# print("ENC :: note, time : ", note.shape, time.shape)


# t_array = dataset.snote_time_to_tarray(note, time)
# print("t_array : ", t_array)
# ns_ = dataset.tarray_to_ns(t_arr= t_array, instruments= ['choir aahs'])

# m_path = 'F:\\rushikesh\\project\\polymuse-future\\midis\\' + model_name + " _NT1_" + "gsF_512_m_oactave_v2_s_1___b_128_e_200_d_0.3__" + "gTsF_512_m_oactave_tm_v1___b_64_e_200_d_0.3.mid"

# dataset.ns_to_midi(ns_, m_path)


"""
Hirachichal Model 
"""

# ar1, ar2 = d2.sync_tarray(ar[:1], ar[1:2])
# print(ar1, ar2)




# ip_memory = 32



# tm = dataset.ns_tarray_to_time(ar1)
# print("tm : ", tm.shape)
# tm[tm > 31] = 31

# tm = dutils.trim_axis2(tm)
# tm = enc_deco.tm_to_enc_tm(tm)
# print('tm : ', tm.shape)
# x_t, y_t = dataset.prepare_time_data(tm, enc_shape = (64, ), ip_memory= 32)
# print('x, y : ', x_t.shape, y_t.shape)
# x_t , y_t = x_t[0], y_t[0]
# # print(x, '--x time')
# print('x, y : ', x_t.shape, y_t.shape)


# DEPTH = 1

# sroll_note = dataset.ns_tarray_to_sFlat(ar1, DEPTH)

# print("sroll_note : ", sroll_note.shape, " --- sroll_note")

# enc = enc_deco.sFlat_to_octave(sroll_note)
# print('enc : ', enc.shape)
# x_nt, y_nt = dataset.prepare_sFlat_data(enc, enc_shape=(2, 16), ip_memory=32, depth = DEPTH)

# print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)
# x_nt, y_nt = x_nt[0, : , :, :1], y_nt[0, :, :1]
# print('x_nt, y_nt : ', x_nt.shape, y_nt.shape)

# print("\n")

# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



# DEPTH = 2
# sroll_drum = dataset.ns_tarray_to_sFlat(ar2, DEPTH)

# print("sroll_drum: ", sroll_drum.shape, " --- sroll_drum")

# enc = enc_deco.sFlat_to_octave(sroll_drum)
# print('enc : ', enc.shape)
# x_dr, y_dr = dataset.prepare_sFlat_data(enc, enc_shape=(2, 16), ip_memory=32, depth = DEPTH)
# x = x_dr
# print('x_dr, y_dr : ', x_dr.shape, y_dr.shape)
# x_dr, y_dr = x_dr[0, : , :, :2], y_dr[0, :, :2]
# print('x_dr, y_dr : ', x_dr.shape, y_dr.shape)

# note = "F:\\rushikesh\\project\\polymuse-future\\h5_models\\lead\\gsF_512_m_oct_dataset_v1___b_64_e_200_d_0.3.h5"

# drum = "F:\\rushikesh\\project\\polymuse-future\\h5_models\\drum\\gsF_512_m_oct_drum_adam_track_s2_v9___b_32_e_100_d_0.3.h5"

# # note = rnn.load_model(note)
# # drum = rnn.load_model(drum)
# lr = 0.0015
# model_name = "dense_drum_note_adam"

# batch_size = 32

# dense_count = 96
# epochs = 300

# # drnm_1, drnm_2 = rnn.drum_note_h_dense(note, drum, x_nt, y_nt, x_dr, y_dr, model_name=model_name, dense_count = dense_count, epochs=epochs, lr = lr)

                                                        # fn1 = 'F:\\rushikesh\\project\\polymuse-future\\hist\\dense3\\gDnDF_1_h_' + str(dense_count)+ "_lr_" +  str(lr)  + '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + ".json"
                                                        # fn2 = 'F:\\rushikesh\\project\\polymuse-future\\hist\\dense3\\gDnDF_2_h_' + str(dense_count)+ "_lr_" +  str(lr)  + '_m_' + model_name+'__b_' + str(batch_size) + "_e_"+str(epochs) + ".json"

                                                        # # drawer.draw_json_loss_acc(fn1, fn2)

# ini_ip, ini_t, ini_drm_ip, ini_drm_t = numpy.zeros(x_nt[0].shape), numpy.zeros(x_t[0].shape), numpy.zeros(x[0, 0].shape), numpy.zeros(x[0].shape)

# # print(ini_ip.shape, ini_t.shape, ini_drm_ip.shape, ini_drm_t.shape, "-- ini_ip, ini_t, ini_drm_ip, ini_drm_t")

# mod = rnn.load_piano_drum_dense_models()

# # print(mod)

# mn, mt, md =  rnn_player.rnn_dense_player(mod, (ini_ip, ini_t, ini_drm_ip, ini_drm_t), ip_memory= 32, predict_instances= 200)

# print(mn, mt, md, "-- mn, mt, md")

# print(mn.shape, mt.shape, md.shape, "-- mn.shape, mt.shape, md.shape")

# tarr = multi_track.dual_pianodrum_rollsto_tarray(mn, mt, md)

# ns_r = dataset.tarray_to_ns(tarr, instruments=['piano', 'guitar'], drm=1)

# f_mid = 'F:\\rushikesh\\project\\polymuse-future\\midis\\' + model_name + '_NT_dual_trk.mid'

# mid = dataset.ns_to_midi(ns_r, f_mid)


"""
BUilder Testing
"""
# # f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\Drummer_Piano.mid'
# # f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\Fur_Elise_by_Ludwig_Van_Beethoven.mid'
# f = 'F:\\rushikesh\\project\\polymuse-future\\midis\\desp.mid'



# shapes = ((32 ,32, 1, 2, 16), (32, 32, 64))
# # mnote, mtime = 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\piano\\stateful\\gsF_512_m_note__b_32_e_40_d_0.3.h5', 'F:\\rushikesh\\project\\polymuse-future\\h5_models\\piano\\stateful\\gTsF_512_m_note__b_32_e_40_d_0.3.h5'
# m_note, m_time, shapes, inp = builder.build_piano_stateful_model(f, epochs=40)
# # m_note, m_time = rnn.load(mnote), rnn.load(mtime)
# player.play_piano_stateful_model_single_track(m_note, m_time, shapes, midi_path='c:/Users/rushi/OneDrive/Desktop/for_fun_midi2.mid', inp = inp, instruments = ['guitar'] )

"""
testing for huge dataset
through the generator
"""
# DST = 'F:/rushikesh/project/dataset/lakh_dataset'
# fs = dutils.get_all_midis(DST, maxx= 20)



# data_gen = data_generator.NoteDataGenerator(0, fs, 32, 32)

# # for d in range(10):
# #     print(data_gen.__getitem__(d))

# print(data_gen)
# rnn_gpu.build_sFlat_model(data_gen, epochs=50, dev = True, cell_count= 512)
# print(data_gen)

"""
train.py ...
"""
# F = "F:\\rushikesh\\project\\dataset\\lakh_dataset"
# train.train_gpu(F, 5)

'''
Player testing
'''
# F = "F:\\rushikesh\\project\\dataset\\lakh_dataset\\Kenny G"
# F = dutils.get_all_files(F)[0]
# #  midi_file = 'midi152.mid' # midi file with at least 3 tracks

# player.play_3_track_no_time(F, midi_fname = 'midiout00')
# player.mid_to_wav('./default699.mid')

"""
Track player 
"""
# F = "F:\\rushikesh\\project\\dataset\\lakh_dataset\\Kenny G"
# F = dutils.get_all_files(F)[0]

# player.play_on_3_track_no_time(F, midi_fname='midiout11')

"""
******** COLLAB ********
"""
# DATASET = 'F:\\rushikesh\project\dataset\lakh_dataset'
# train.train_gpu(DATASET, maxx = 15, epochs = 100)


# midi_file = 'I Only Want to Be With You.mid' # midi file with at least 3 tracks
# midi_file = 'midi492.mid' # midi file with at least 3 tracks

# player.play_on_3_track_no_time(midi_file, midi_fname = 'midiout00')

"""
Image Representation
"""
# F = 'F:\\rushikesh\project\data2'
# F = 'F:\\rushikesh\project\dataset\lakh_dataset'
# midis = dutils.get_all_midis(F, maxx = 100)
# print(len(midis))
# print(midis)
# datagen = data_generator.NoteDataGenerator(0, midis, 32, 32, enc= False)

# for i in range(10):
#     x = datagen.__getitem__(i)
#     print(x[0].shape, x[1].shape)

# L = [[] for i in range(32)]
# print(L, len(L))
# import cv2
# IMG = numpy.zeros((32, 128, 128), dtype = 'uint8')
# for i in range(datagen.steps_per_epoch):
#     x, y = datagen.__getitem__(i)
#     # print(x.shape, y.shape, " >>>>>>>>>>>>>>>>>>")
#     for j in range(x.shape[0]): # note instances iterator
#         for k in range(x.shape[1]): # ip memmory iterator
#             for d in range(x.shape[2]):
#                 # print(j, k, d)
#                 # print( y.shape, " //////////////////")
#                 # print(y[k, d], "********************")
#                 # print(int(x[j, k, d]), " -=-=-=-=-=-=-=")
#                 # L[k].append([int(x[j, k, d]), y[j, d]])
#                 xn, yn = int(x[j, k, d]), int(y[j, d])
#                 IMG[k, xn, yn] += 1

# Mx = numpy.max(IMG)
# print(Mx, numpy.unique(IMG))

# for i in range(IMG.shape[0]):
#     IMG[i] = numpy.interp(IMG[i], (IMG[i].min(), IMG[i].max()), (1, 255))

# print(IMG.max(), IMG.min(), " - - - - - - - ")

# KEN = numpy.full((IMG[i].shape), 4294967295)

# # IMG = IMG.astype(int)
# MGH = copy.deepcopy(IMG[0])
# kernel = numpy.ones((5,5), numpy.uint8)
# for i in range(32):
#     # img_dilation = cv2.dilate(IMG[i], kernel)
#     img = IMG[i]
#     if i != 0: 
#         MGH = MGH * IMG[i]
#         MGH = numpy.bitwise_and(MGH, KEN)
#     # print(img.shape, ' -- shape')
#     scale_percent = 700 # percent of original size
#     width = int(img.shape[1] * scale_percent / 100)
#     height = int(img.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     # resize image
#     img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#     cv2.imwrite("./img/" + str(i) + ".jpg", img) 
#     # for j in range(128):
#     #     yp = numpy.where(IMG[i, j] > 0)[0]
#     #     # print(yp)
#     #     xp = numpy.full(yp.shape, j)
#     #     plt.scatter(xp, yp)
#     # plt.savefig("./img/" + str(i) + ".png")

# MGH = numpy.interp(MGH, (MGH.min(), MGH.max()), (1, 255))



# scale_percent = 700 # percent of original size
# width = int(MGH.shape[1] * scale_percent / 100)
# height = int(MGH.shape[0] * scale_percent / 100)
# dim = (width, height)
# # resize image
# MGH = cv2.resize(MGH, dim, interpolation = cv2.INTER_AREA)
# cv2.imwrite("./img/i"  + ".jpg", MGH) 
# print(len(L), len(L[0]), "-- L")
# L = numpy.array(L)
# print(L.shape, "-- L")

"""
Transpose the dataset
"""
# import music21
# import sys
# F = 'F:\\rushikesh\project\dataset\lakh_dataset'
# midis = dutils.get_all_midis(F, maxx = 1300)
# print(len(midis))

# # major conversions
# majors1 = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("G-", 6),("G", 5)])
# minors1 = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("G-", 3),("G", 2)])

# # major conversions
# majors = dict([("G#", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("C#", -1),("D", -2),("D#", -3),("E", -4),("F", -5),("F#", 6),("G", 5)])
# minors = dict([("G#", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("C#", -4),("D", -5),("D#", 6),("E", 5),("F", 4),("F#", 3),("G", 2)])

# minors.update(minors1)
# majors.update(majors1)

# print(minors, "\n", majors)

# DT = 'F:/rushikesh/project/data2/'
# with open('trkfile.csv', 'r') as ff:
#     tfilestr = ff.read()

# print(tfilestr) 
# sett = set([f for f in tfilestr.split('\n')])
# # print("settt : ", sett)
# for m in midis:
#     if m in sett: continue
#     try:
#         print(m)
#         score = music21.converter.parse(m)
#         key = score.analyze('key')
#     #    print key.tonic.name, key.mode
#         if key.mode == "major":
#             halfSteps = majors[key.tonic.name]
            
#         elif key.mode == "minor":
#             halfSteps = minors[key.tonic.name]
        
#         newscore = score.transpose(halfSteps)
#         key = newscore.analyze('key')
#         print (key.tonic.name, key.mode)
#         newFileName = DT +  "C_" + m.split("\\")[-1]
#         print(newFileName)
#         newscore.write('midi',newFileName)
#         sett.add(m)
#     except Exception as e:
#         print(e) 
#         continue

# with open('trkfile.csv', 'w') as ff:
#     ff.write('\n'.join(sett))



'''
load
'''
# loader.load_midi()
"""
Load new models : 
"""
# ip_memory = 32
# DST = 'F:/rushikesh/project/polymuse-future/h5_models/lead/stateless/'

# DST_F = DST + os.listdir(DST)[0]

# print('DST 45 : ', DST_F)

# model = rnn_gpu.load(DST_F)

# print(model)

# ini = numpy.array([random.choice(pattern.ip_patterns)])

# # ini = numpy.array([ini, ini, ini])
# ini = numpy.append(ini, ini)
# ini = numpy.array([ini])
# print("0000000000000000000 : ini : ", ini.shape)
# ini = d2.ip_patterns_to_octave(ini)
# print('ininininini : ', ini.shape)
# ini, y = dataset.prepare_sFlat_data(ini)
# ini = ini[:, 3]

# print('ini : ', ini.shape)
# ini = ini[:, : , 0]
# # ini = numpy.append(ini, ini[:, :, 1])
# ini = numpy.reshape(ini, (1, 32, 1, 2, 16))
# print('ini : ', ini.shape)


# note, time = rnn_player.rnote_player(model, ini= ini[0])

# t_array = dataset.snote_time_to_tarray(note, None, deltam= 8)

# print("t_array : ", t_array.shape)
# ns_ = dataset.tarray_to_ns(t_arr= t_array, instruments= ['piano'])

# m_path = 'F:\\rushikesh\\project\\polymuse-future\\midis\\' + 'te' + 'e50zjut2' + '.mid'

# dataset.ns_to_midi(ns_, m_path)


