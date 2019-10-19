from polymuse import dataset, transformer, enc_deco, dutils, dataset2 as d2
from polymuse import multi_track


from polymuse import rnn_player
from polymuse import drawer


import numpy, os


# def play_piano_model_single_track(m_note, m_time, shapes, model_name = 'note_time', midi_path = None, x_nt = None, x_t = None, y_nt = None, y_t = None, ip_memory = 32, instruments = ['piano']):    
#     ip_nt = numpy.zeros(shapes[0])
#     ip = numpy.zeros(shapes[1])

#     print('ip_nt, ip : ', ip_nt.shape, ip.shape)

#     note, time = rnn_player.rsingle_note_time_play(m_note, m_time, ip_nt, ip, y_nt, y_t, ip_memory, 350)

#     print("note, time : ", note.shape, time.shape)

#     note, time = enc_deco.octave_to_sFlat(note), enc_deco.enc_tm_to_tm(time)

#     print("ENC :: note, time : ", note.shape, time.shape)

#     t_array = dataset.snote_time_to_tarray(note, time)
#     print("t_array : ", t_array)
#     ns_ = dataset.tarray_to_ns(t_arr= t_array, instruments= instruments)

#     m_path = 'F:\\rushikesh\\project\\polymuse-future\\midis\\' + model_name + " _NT1_" + "gsF_512_m_oactave_v2_s_1___b_128_e_200_d_0.3__" + "gTsF_512_m_oactave_tm_v1___b_64_e_200_d_0.3.mid"
#     m_path = midi_path if midi_path else m_path
#     dataset.ns_to_midi(ns_, m_path)


# def play_piano_stateful_model_single_track(m_note, m_time, shapes, model_name = 'note_time', midi_path = None, inp = None ,x_nt = None, x_t = None, y_nt = None, y_t = None, ip_memory = 32, instruments = ['piano']):    
#     if not inp:
#         ip_nt = numpy.zeros(shapes[0])
#         ip = numpy.zeros(shapes[1])
#     else: 
#         ip_nt = numpy.array(inp[0][:32])
#         ip = numpy.array(inp[1][:32])

#     print('ip_nt, ip : ', ip_nt.shape, ip.shape)

#     note, time = rnn_player.rsingle_note_time_stateful_play(m_note, m_time, ip_nt, ip, y_nt, y_t, ip_memory, 350)

#     print("note, time : ", note.shape, time.shape)

#     note, time = enc_deco.octave_to_sFlat(note), enc_deco.enc_tm_to_tm(time)

#     print("ENC :: note, time : ", note.shape, time.shape)

#     t_array = dataset.snote_time_to_tarray(note, time)
#     print("t_array : ", t_array)
#     ns_ = dataset.tarray_to_ns(t_arr= t_array, instruments= instruments)

#     m_path = 'F:\\rushikesh\\project\\polymuse-future\\midis\\' + model_name + " _NT1_" + "gsF_512_m_oactave_v2_s_1___b_128_e_200_d_0.3__" + "gTsF_512_m_oactave_tm_v1___b_64_e_200_d_0.3.mid"
#     m_path = midi_path if midi_path else m_path
#     dataset.ns_to_midi(ns_, m_path)


def play_single_track():
    pass

def play_3_track():
    pass

def play_single_track_no_time(input_file):
    if not os.path.isfile(input_file): raise FileNotFoundError
    pass