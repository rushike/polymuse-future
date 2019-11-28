from polymuse import dataset, transformer, enc_deco, dutils, dataset2 as d2, evaluation, constant, data_generator, rnn_h, rnn_gpu, pattern, rnn_player2
# from polymuse import multi_track


from polymuse import train, player, loader
# from polymuse import rnn_player
from polymuse import drawer

from matplotlib import pyplot as plt

# from polymuse import builder, player
# from polymuse import pattern

# from scipy.interpolate import make_interp_spline, BSpline
import numpy, random, pprint, copy

import gc, os
import warnings
warnings.filterwarnings("ignore")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


"""
Stateful Note Data generator
"""
# loader.load()

F = 'F:\\rushikesh\project\dataset\lakh_dataset'
# F = 'F:\\rushikesh\project\polymuse-future\single_song'
M = 'F:\\rushikesh\project\dataset\lakh_dataset\A_Teens\Mamma Mia.mid'
# M = 'F:\\rushikesh\project\polymuse-future\single_song\Single_Test_Solo.mid'
# M1 = 'F:\\rushikesh\project\polymuse-future\yyy1145.mid'
# M2 = 'F:\\rushikesh\project\polymuse-future\yyy224.mid'

# C = 'F:\\rushikesh\project\dataset\lakh_dataset\Atlantic Starr\Secret Lovers.mid'
# ns = dataset.to_note_sequence('F:\\rushikesh\\project\\polymuse-future\\single_song\\Single_Test_Solo.mid')
# print(ns)\
# fs = dutils.get_all_midis(F, maxx= 10)
# data_gen = data_generator.NoteDataGenerator(2, fs, 256, 32)
# print(data_gen)
# print(data_gen.__getitem__(0))
# train.train_gpu(F,maxx=20, epochs= 10)
# train.train_stateful_gpu(F, epochs= 50) # Single song
# player.play_statefull_3track(M)

# player.play_3_track_no_time(M, predict_instances=400)

# player.play_on_3_track_no_time(M, predict_instances=400)


"""
Testing NoteTimeGenerator
"""
# F = 'F:\\rushikesh\project\dataset\lakh_dataset'
# F = 'F:\\rushikesh\project\polymuse-future\single_song'
# M = 'F:\\rushikesh\project\dataset\lakh_dataset\A_Teens\Mamma Mia.mid'
F = 'F:\\rushikesh\project\dataset\lakh_dataset\A_Teens'
# M = 'F:\\rushikesh\project\polymuse-future\single_song\Single_Test_Solo.mid'
fs = dutils.get_all_midis(F, maxx= 100)
fs = random.choices(fs, k=2)
print(fs)
g0 = data_generator.DataGenerator_3Tracks(fs, 32,128, test_size = 0.5)
# g1 = data_generator.NoteTimeDataGenerator(1, fs, 32,128)
# g2 = data_generator.NoteTimeDataGenerator(2, fs, 32,128)
print(g0.train.steps_per_epoch, g0.val.steps_per_epoch)

# rnn_gpu.build_sflattimebin_funct(gens= g0, epochs = 10)

# model = rnn_gpu.load('./h5_models/vuuy.h5')
# losses = {
# 	    "lead_track": rmsecat(1),
# 	    "chorus_track": rmsecat(3),
#         "drum_track" : rmsecat(2)
#     }

model = rnn_gpu.load_model('./h5_models/vuuy.h5', custom_objects= {'rmsecat_' : rnn_gpu.rmsecat(2)})

print("model type : ", type(model))
x, y = g0.train.__getitem__(0)
# print(y, y[0].shape, y[1].shape, y[2].shape)
t_array = rnn_player2.polymuse_player(model, x, predict_instances=600)

mid_path = './' + "out" + str(random.randint(0, 1000)) + '.mid'
instruments = ['piano', 'guitar','choir aahs']
ns_ = dataset.tarray_to_ns(t_arr= t_array, instruments= instruments, drm = 2)
print("midi file : ", mid_path)
dataset.ns_to_midi(ns_, mid_path)
