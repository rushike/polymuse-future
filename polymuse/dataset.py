from polymuse import constant

from magenta.music import midi_io
# from magenta.protobuf.music_pb2 import NoteSequence
from magenta.protobuf import music_pb2
import numpy, copy, sys, traceback

"""
It includes the basic fuctionlity of input in and out, from midi to one of representation used in polymuse
Hierarchy of  Representation : 
    -- midi 
    -- ns (Note Sequence) --> magenta representation
    -- tarray : polymuse top level representaion
    -- sFlat, time : encoded format (top level encoding)
    -- octave, time : model encoded format

"""

def to_note_sequence(midi_file_path):
    return midi_io.midi_file_to_note_sequence(midi_file_path)

def to_pretty_midi(midi_file_path):
    return midi_io.pretty_midi.PrettyMIDI(midi_file_path)

def pretty_midi_to_ns(pretty_midi):
    return midi_io.midi_to_note_sequence(pretty_midi)

def to_array(midi_file_path, seconds = False):
    # print(midi_file_path, type(midi_file_path))
    if isinstance(midi_file_path, str):
        ns = to_note_sequence(midi_file_path)
        tempo = ns.tempos[0].qpm
    elif isinstance(midi_file_path, music_pb2.NoteSequence) :
        ns = midi_file_path
        tempo = ns.tempos[0].qpm
    if isinstance(midi_file_path, list):
        notes = midi_file_path
        tempo = 120
    else : 
        # ns = midi_file_path
        # tempo = 120
        notes = ns.notes
    # print("+++++++++++++++++++++++++++++++++ ", tempo)
    N = len(notes)
    res = numpy.zeros((N, 4))

    MULT = 1 if seconds else tempo / 60 * 4

    """
    Format [abstime, pitch, velocity, duration] 
    duration in 0 - 32 format
    """
    prev_time = 0
    for i, n in enumerate(notes):
        res[i] = [n.start_time, n.pitch, n.velocity, numpy.round(32 * (((n.end_time - n.start_time) * MULT) ))]
    
    return res

def to_notes(midi_file_path):
    res = to_notes(midi_file_path)
    return res[:, 1]

def to_time(midi_file_path):
    res = to_notes(midi_file_path)
    return res[:, 3]

def to_velo(midi_file_path):
    res = to_notes(midi_file_path)
    return res[:, 2]

def get_instrument(ns, inst, len_adjuster = 1, resolution = 32):
    return list(filter(lambda x: x.instrument == inst, ns.notes))

def get_instrument_array(ns, inst, len_adjuster = 1, resolution = 32):
    return ns_list_to_array(get_instrument(ns, inst), len_adjuster, resolution)

def ns_list_to_array(li, len_adjuster = 1, resolution = 32):
    # if not isinstance(li, Note): return "NOT TRUE"
    # if isinstance(li, 'int') :li = get_instrument(ns, inst)
    N = len(li)
    res = numpy.zeros((N, 4))
    for i, n in enumerate(li):
        res[i] = [numpy.round(resolution * n.start_time * len_adjuster), n.pitch, n.velocity, numpy.round(resolution * ((n.end_time - n.start_time) * len_adjuster))]
    return res

def ns_to_tarray(ns, resolution = 32, seconds = False):
    listn = []
    MAX = 16
    MULT = 1 if seconds else ns.tempos[0].qpm / (60  *4)
    # print(MULT)
    lmax = 0
    for inst in range(MAX):
        ins_array = get_instrument_array(ns, inst, len_adjuster = MULT, resolution= resolution)
        lmax = len(ins_array) if len(ins_array) > lmax else lmax
        if not ins_array.any(): continue
        listn.append(ins_array)
    le = len(listn)
    numar = numpy.zeros((le, lmax, 4))
    for i in range(le):
        for j in range(len(listn[i])):
            numar[i, j] = listn[i][j]
    # fo
    # numar = numpy.array(listn)
    return numar

def merge_to_3tracks(ns):
    for nt in ns.notes:
        if nt.is_drum: 
            nt.instrument = 3
        elif nt.program in constant.lead_track:
            nt.instrument = 1
            nt.program = 1
        else:
            nt.instrument = 2
            nt.program = 25
    return ns

def merge_tarray(t_arr, lead_arr):
    # lead_arr, t_arr = t_arr, lead_arr if len(t_arr) > len(lead_arr) else lead_arr, t_arr #ct_arr > t_arr
    # for i in range(t_arr.shape[0]):
    #     t_p, ct_p = 0, 0
    #     while True:
            
    #         pass
    # print("t_arr : ", t_arr.shape, ", lead_arr : ", lead_arr.shape)
    # print(t_arr[0, : 5])
    # print(lead_arr[0, : 5])
    l_arr = []
    for i in range(t_arr.shape[0]):
        t_list = t_arr[i].tolist()
        lead_list = lead_arr[i].tolist()

        lead_list.extend(t_list)
        # print(lead_list, ">>>")
        # print(t_list, "---")
        lis = sorted(lead_list, key=lambda x: x[0] if x[0] != 0 else sys.maxsize)

        # l_arr = numpy.array(lis)
        l_arr.append(lis)
    l_arr = numpy.array(l_arr)
    # print(l_arr[0, :5], l_arr.shape, " -- l_arr")
    return l_arr

def default_ns():
    ns = music_pb2.NoteSequence()
    # Populate header.
    ns.ticks_per_quarter = 480
    ns.source_info.parser = music_pb2.NoteSequence.SourceInfo.PRETTY_MIDI
    ns.source_info.encoding_type = (music_pb2.NoteSequence.SourceInfo.MIDI)

    # Populate time signatures.
    time_signature = ns.time_signatures.add()
    time_signature.time = 0 
    time_signature.numerator = 4
    time_signature.denominator = 4

    # Populate key signatures
    key_signature = ns.key_signatures.add()
    key_signature.time = 0
    key_signature.key = 0
    midi_mode = 0

    # Populate tempo changes
    tempo = ns.tempos.add()
    tempo.time = 0
    tempo.qpm = 120

    # Populate instrument name from the midi's instruments
    instrument_info = ns.instrument_infos.add()
    instrument_info.name = 'Guitar'
    instrument_info.instrument = 25 #nylon guitar

    return ns




def tarray_to_ns(t_arr, instruments = None, zero_based = True, drm = None):
    ns = default_ns()
    timmer = 0
    delta = 0
    d_drk = [False for _ in range(t_arr.shape[0])]
    if drm: d_drk[drm] = True
    print(d_drk, "--drk")
    instruments = instruments if instruments else ['piano'] * t_arr.shape[0]
    instruments = constant.instrument_program_code(instruments)
    print(t_arr.shape)
    print("instruments : ", instruments)
    MULT = (ns.tempos[0].qpm / (60 * 4) ) * 32
    for i in range(t_arr.shape[0]):
        timmer = 0
        for j in range(t_arr.shape[1]):
            if t_arr[i, j, 3] == 0 and zero_based: continue
            delta = t_arr[i, j, 3] / MULT if t_arr[i, j, 0] != 0 else 0
            if t_arr[i, j, 1] == 0 or t_arr[i, j, 2] == 0:
                timmer += delta
            note = ns.notes.add()
            note.instrument = i
            if not d_drk[i] : note.program = instruments[i]
            note.is_drum = d_drk[i]
            note.start_time =  timmer
            timmer += delta
            
            note.end_time = timmer 
            
            note.pitch = int(t_arr[i, j, 1])
            note.velocity = int(t_arr[i, j, 2])
            # note.is_drum = False
    return ns

def ns_to_midi(ns, file_path):
    midi_io.note_sequence_to_midi_file(ns, file_path)
    return True

def ns_to_pretty_midi(ns):
    return midi_io.note_sequence_to_pretty_midi(ns)
    
def ns_tarray_to_time(t_arr):
    cp_t_arr = numpy.zeros(t_arr[:, :, 3:4].shape)
    tm = 0
    for i in range(t_arr.shape[0]):
        for j in range(1, t_arr.shape[1]):
            if t_arr[i, j - 1, 0] == t_arr[i, j, 0]: 
                # print("repeat")
                continue
            cp_t_arr[i, tm, 0] = t_arr[i, j, 3]
            tm += 1
    cp_t_arr = cp_t_arr[:, :, : tm + 1]
    cp_t_arr = cp_t_arr[~numpy.all(cp_t_arr == 0, axis=2)]
    return numpy.reshape(cp_t_arr, (1,) + cp_t_arr.shape)

def ns_tarray_to_sFlat(t_arr, DEPTH = 2):
    """It is only the note representaion of sequence
    ### Verified
    Arguments:
        t_arr {ns_tarray-->numpy.ndarray} -- [description]
        SPREAD {[int]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    sflat = []
    tm, tm_max, dep = 0, 0, 0
    lis = t_arr.tolist()
    for i in range(t_arr.shape[0]):
        ctime, aflag = -1, False
        sflat.append([])
        sflat[i].append([0 for _ in range(DEPTH)])
        
        tm, dep = -1, 0 # tm -1 initiator
        
        st_arr = sorted(lis[i], key=lambda x: x[0] if x[0] != 0 else sys.maxsize)

        # print(st_arr[:10], t_arr[:10])
        t_arr = numpy.array([st_arr])
        # print(st_arr)
        # print(st_arr[:10], t_arr[:10])


        for j in range(t_arr.shape[1]):
            if dep == DEPTH: 
                dep = 0
                # tm += 1
                # sflat[i].append([0 for _ in range(SPREAD)])
            if ctime == t_arr[i, j, 0] and dep == 0: 
                continue
            elif ctime != t_arr[i, j, 0]:
                sflat[i].append([0 for _ in range(DEPTH)])
                tm += 1
                dep = 0

            sflat[i][tm][dep] = int(t_arr[i, j, 1])
            ctime = t_arr[i, j, 0]
            dep += 1
             
    tm_max = max([len(v) for v in sflat])
    
    # print("sfffff : ", sflat)
    # print("tm____max_________________________ : ", tm_max)
    # print(sflat, "--sflat")

    #triming the array
    sflat_arr = numpy.zeros((t_arr.shape[0], tm_max + 1, DEPTH), dtype = 'int32') #because tm_max is zeo based indexing
    for i in range(sflat_arr.shape[0]):
        for j in range(len(sflat[i])):
            for k in range(sflat_arr.shape[2]):
                sflat_arr[i, j, k] = sflat[i][j][k]
    return sflat_arr

# def tarray_to_sFlat_roll(t_arr, DEPTH = 3):
#     roll = []
#     lis = t_arr.tolist()
#     # print(t_arr.shape)
#     for i in range(t_arr.shape[0]):
#         le = t_arr.shape[1] 
#         # blen = 128
#         broll = numpy.zeros((128, DEPTH))  #total 128 sFlat Time instances ------------> list syntax : [[0 for _ in range(DEPTH)] for _ in range(128)]
#         roll.append([])
        
#         p_time, dep = -1, 0 # tm -1 initiator
        
#         leftover = 0 #numpy.zeros((DEPTH), dtype= 'int32')
#         blen = 0 #* numpy.ones((DEPTH), dtype= 'int32')

#         st_arr = sorted(lis[i], key=lambda x: x[0] if x[0] != 0 and x[1] == 0 else sys.maxsize)
#         st_arr = numpy.array(st_arr)
#         # print(st_arr, "")
#         for j in range(le):
            
#             timest, note, velo, dura = (int(v) for v in st_arr[j])

#             if dep == DEPTH: 
#                 dep = 0
#             if p_time == timest and dep == 0: 
#                 continue
#             elif p_time == timest:
#                 # if leftover[dep - 1] == 0: blen[dep - 1] += dura
#                 # else : blen += leftover[dep - 1]
#                 blen -= dura
#                 pass
#             elif p_time != timest:
#                 dep = 0
                
#             if blen + dura >= 128:
#                 leftover = abs(128 - blen - dura)
#             else : leftover = 0

#             if blen > 127: 
#                 roll[i].append(broll.tolist())
#                 broll = numpy.zeros((128, DEPTH))  #total 128 sFlat Time instances, 4 MEASURES ------------> list syntax : [[0 for _ in range(DEPTH)] for _ in range(128)]
#                 blen = 0 
#                 dep = 0
                 
#                 #Dealing with leftover ... 
#                 # if leftover.any():
#                 #     for ind, l in enumerate(leftover):
#                 #         broll[blen - l : blen, ind] = note
#                 #         leftover[ind] = 0
#                 #     blen -= dura
            
#             # print(timest, note, velo, dura)
            
#             broll[blen : blen + dura - leftover - 1, dep] = note
#             broll[blen + dura - leftover - 1, dep] = 0
#             blen += dura
#             dep += 1
#             p_time = timest
#             broll    
#         # print(broll.tolist(), broll.shape, "--broll")
#         # print(roll, "---roll")
#     roll[i].append(broll.tolist())
#     print(broll.tolist())

#     # print(roll, "---roll")
    
    
#     ar_roll = numpy.array(roll)
#     le = max([len(a) for a in ar_roll])
#     ar_roll = numpy.zeros((ar_roll.shape[0], le, 128, DEPTH))
#     for i in range(ar_roll.shape[0]):
#         ln = len(roll[i])
#         for j in range(ln):
#             ar_roll[i, j] = roll[i][j]
#     print(ar_roll, ar_roll.shape, "--roll shape") 
#     # roll = numpy.roll(ar_roll, 128 - blen, axis = 1)   
#     return roll

def to_pianoroll(pretty_midi, fs):
    p_list = []
    p, q, r = 0, 0, 0
    for i, ins in enumerate(pretty_midi.instruments):
        roll = ins.get_piano_roll(fs)
        p_list.append(roll.tolist())
        p += 1
        q = max([q, roll.shape[0]])
        r = max([r, roll.shape[1]])
        # print(p_list)
    
    res = numpy.zeros((p, r, q), dtype = 'int32')
    for i in range(len(p_list)):
        for j in range(len(p_list[i])):
            for k in range(len(p_list[i][j])):
                res[i, k, j] = p_list[i][j][k]
        
    return res

def pianoroll_to_sFlat_roll(roll, DEPTH):
    sflatroll = numpy.zeros(roll.shape[:-1] + (DEPTH, ))
    for i in range(sflatroll.shape[0]):
        for j in range(sflatroll.shape[1]):
            c = numpy.where(roll[i, j] > 0)[0]
            bound = min([c.shape[0], DEPTH])
            if bound!= 0:
                sflatroll[i, j, :bound] = c[:bound]
    return sflatroll

def note_time_to_tarray(note, time):
    tarr = numpy.zeros((1, note.shape[1], 4))
    timmer = 0
    delta = 0
    for i in range(tarr.shape[0]):
        for j in range(tarr.shape[1]):
            delta = time[i, j]
            tarr[i, j, 0] = timmer
            tarr[i, j, 1] = note[i, j]
            tarr[i, j, 2] = 80
            tarr[i, j, 3] = delta
            timmer += delta
    # print(tarr, tarr.shape)
    return tarr

def snote_time_to_tarray(note, time, deltam = None, velo = 80):
    tarr = numpy.zeros((1, note.shape[1] * note.shape[2], 4))
    timmer = 0
    delta = 0
    tp = 0
    for i in range(tarr.shape[0]):
        for j in range(note.shape[1]):
            for k in range(note.shape[2]):
                if note[i, j, k] == 0: continue
                delta = time[i, j] if not deltam else deltam
                tarr[i, tp, 0] = timmer
                tarr[i, tp, 1] = note[i, j, k]
                tarr[i, tp, 2] = velo
                tarr[i, tp, 3] = delta 
                tp += 1
            timmer += delta
    return tarr


def add_lead_in_sFlat(sroll, lead_roll):
    # print("----- sroll ", sroll.shape)
    res_roll = numpy.zeros((sroll.shape[:-1] + (sroll.shape[-1] + 1, )))
    # print("res sroll : ", res_roll.shape)
    # # res_roll[:, :, 1:] = sroll[:, :, :]
    # # res_roll[:, :, 0] = lead_roll[:, :, 0]
    # print("lead shape : ", lead_roll.shape)
    if sroll.shape[0] < lead_roll.shape[0]:
        lead_roll = lead_roll[:sroll.shape[0]]
    for i in range(lead_roll.shape[0]):
        if sroll.shape[1] < lead_roll.shape[1]:
            lead_roll = lead_roll[:, :sroll.shape[1]]
        for j in range(lead_roll.shape[1]):
            res_roll[i, j, 1:] = sroll[i, j]
            res_roll[i, j, 0] = lead_roll[i, j, 0]
    return res_roll


def prepare_sFlat_data(notes, track_range = None, enc_shape = (2, 16), ip_memory = 32, depth = 2, spread = 16):
        """Prepares data for the network(RNN) in ip/op format. Here called data_in, data_out.
        With so callled vocab_size of ip_memory
        
        Arguments:
            notes {sFlat__encoded -> numpy.ndarray} -- [description]
        
        Keyword Arguments:
            track_range {[type]} -- [description] (default: {None})
            enc_shpae {tuple} -- [description] (default: {(2, 16)})
            ip_memory {int} -- memory or ipsize used in predicting next  (default: {32})
        
        Returns:
            [numpy.ndarray] -- data for network
        """
        track_range = track_range if track_range else [0, 1]
        
        data_in, data_out = [], []
        
        for tr in range(track_range[1] - track_range[0]):
            # trk = tr - track_range[0]
            nt = notes[tr]
            data_in.append([])
            data_out.append([])
            lent = len(notes[tr])
            # for j in range(lent):
            le = len(nt)
                
            chunks_count = le // ip_memory + 1
            
            for i in range(le - ip_memory):
                start, end = i, i + ip_memory
                buf_size = ip_memory if end < le else le -  start # only reason due to logic below else not needed
                buffer = numpy.zeros((ip_memory, depth,) + enc_shape)
                # print(buffer.shape)
                buffer[:buf_size, :] = nt[start : start + buf_size]
                data_in[tr].append(buffer)
                
                data_out[tr].append((nt[end] if end < le else notes[0][0]))
            
        # if track_range[1]- track_range[0] == 1: #is scalar, no track
            # data_in, data_out = data_in[0], data_out[0]
        

        return numpy.array(data_in), numpy.array(data_out)

def prepare_time_data(ntime, track_range = None, enc_shape = (32, ), ip_memory = 32, spread = 32):
    track_range = track_range if track_range else [0, 1]
         
    data_in, data_out = [], []
    
    for tr in range(track_range[1] - track_range[0]):
        # trk = tr - track_range[0]
        nt = ntime[tr]
        data_in.append([])
        data_out.append([])
        lent = len(ntime[tr])
        # for j in range(lent):
        le = len(nt)
            
        chunks_count = le // ip_memory + 1
        
        for i in range(le - ip_memory):
            start, end = i, i + ip_memory
            buf_size = ip_memory if end < le else le -  start # only reason due to logic below else not needed
            buffer = numpy.zeros((ip_memory, ) + enc_shape)
            # print(buffer.shape)
            buffer[:buf_size, :] = nt[start : start + buf_size]
            data_in[tr].append(list(buffer))
            
            data_out[tr].append(list((nt[end] if end < le else ntime[0][0])))
        
    # if track_range[1]- track_range[0] == 1: #is scalar, no track
        # data_in, data_out = data_in[0], data_out[0]

    return numpy.array(data_in), numpy.array(data_out)


def prepare_sFlat_roll_data(sroll, track_range = None, enc_shape = (32, ), ip_memory = 32, depth = 32):
    track_range = track_range if track_range else [0, 1]
         
    data_in, data_out = numpy.zeros((sroll.shape[1] - ip_memory, ip_memory ,sroll.shape[0], sroll.shape[2]) + enc_shape), numpy.zeros((sroll.shape[1] - ip_memory, sroll.shape[0], sroll.shape[2]) + enc_shape)
    
    for tr in range(sroll.shape[0]):
        # trk = tr - track_range[0]
        # for j in range(lent):
        le = sroll.shape[1]
        for i in range(le - ip_memory):
            start, end = i, i + ip_memory
            buf_size = ip_memory if end < le else le -  start # only reason due to logic below else not needed
            # buffer = numpy.zeros((ip_memory, sroll.shape[0],sroll.shape[2]) + enc_shape)
            # print(buffer.shape)
            # buffer[:buf_size, :] = nt[start : start + buf_size]
            data_in[i, :buf_size, tr] = sroll[tr, start : start + buf_size]
            data_out[i, tr] = sroll[tr, start + buf_size]

        
    # if track_range[1]- track_range[0] == 1: #is scalar, no track
        # data_in, data_out = data_in[0], data_out[0]

    return numpy.array(data_in), numpy.array(data_out)
