from polymuse import dataset, transformer, enc_deco, dutils, dataset2 as d2, constant
from polymuse import multi_track, data_generator


from polymuse import rnn_player
from polymuse import drawer


from keras.models import load_model

import numpy, os, random
from midi2audio import FluidSynth

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

def play_3_track_no_time(input_file, raw = False, instruments = ['piano', 'guitar','choir aahs'], midi = True, midi_fname = 'default', wav = False, play = False):
    if not os.path.isfile(input_file): raise FileNotFoundError("Input file specified is not a file : ")
    model_home = './h5_models/'
    st = 'stateless/'
    models = [load_model(dutils.get_all_files(model_home + constant.type3tracks[i])[0]) for i in range(3)]
    print(models)

    
    tarr = []
    for i in range(3):
        x, y = data_generator.note_data(input_file, trk= i, DEPTH= constant.depths_of_3tracks[i])
        print(x.shape, y.shape, "--x , --y")
        note, time = rnn_player.rnote_player(models[i], x, TM = constant.timec_of_3tracks[i], DEPTH= constant.depths_of_3tracks[i])
        # note, time = enc_deco.octave_to_sFlat(note), enc_deco.enc_tm_to_tm(time)
        t_array = dataset.snote_time_to_tarray(note, time)
        tarr.append(t_array)

    if raw: return tuple(tarr)

    mx_tm = max([v.shape[1] for v in tarr])


    t_array = numpy.zeros((3, mx_tm, 4))

    for i in range(3):
        t_array[i, : tarr[i].shape[1]] = tarr[i][0]

    if midi: 
        mid_path = './' + midi_fname + str(random.randint(0, 1000)) + '.mid'
        ns_ = dataset.tarray_to_ns(t_arr= t_array, instruments= instruments, drm = 2)
        dataset.ns_to_midi(ns_, mid_path)
        fs = FluidSynth()
        print(mid_path, " --midi\n", mid_path[:-3] + '.wav', " --wav")
        if play:
            fs.play_midi(mid_path)
        if wav:
            fs.midi_to_audio(mid_path, mid_path[:-3] + '.wav')
        


    
    return t_array

def play_midi(midi_file, fs = FluidSynth()):
    fs.play_midi(midi_file)

def mid_to_wav(midi_file, fs = FluidSynth()):
    print(os.path.isfile(midi_file))
    print(midi_file, " --midi\n", midi_file[:-4] + '.wav', " --wav")
    fs.midi_to_audio(midi_file, midi_file[:-4] + '.wav')

def play_single_track(model, file_path = None, x = None, y = None):
    pass

