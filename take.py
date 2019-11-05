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
Stateful Note Data generator
"""

F = 'F:\\rushikesh\project\dataset\lakh_dataset'
M = 'F:\\rushikesh\project\dataset\lakh_dataset\A_Teens\Mamma Mia.mid'
# train.train_stateful_gpu(F, maxx=10)
# player.play_statefull_3track(M)
player.play_on_3_track_no_time(M, midi_fname='yyy11')