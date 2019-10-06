
from polymuse import constant, dataset, enc_deco

# from magenta.music import midi_io
# from magenta.protobuf.music_pb2 import NoteSequence
# from magenta.protobuf import music_pb2
import numpy, copy, sys


"""
It includes the basic fuctionlity of input in and out, from midi to one of representation used in polymuse
Hierarchy of  Representation : 
    -- midi 
    -- ns (Note Sequence) --> magenta representation
    -- tarray : polymuse top level representaion
    -- sFlat, time : encoded format (top level encoding)
    -- octave, time : model encoded format

It is second file includes same functionality as dataset, the only sole reason to make file is to avoid the uncessary long scroll while development
In last it is  posssibly going to merge with dataset

"""

def tarray_to_sFlat_roll(t_arr, DEPTH = 3):
    roll = []
    lis = t_arr.tolist()
    # print(t_arr.shape)
    for i in range(t_arr.shape[0]):
        le = t_arr.shape[1] 
        # blen = 128
        broll = numpy.zeros((128, DEPTH))  #total 128 sFlat Time instances ------------> list syntax : [[0 for _ in range(DEPTH)] for _ in range(128)]
        roll.append([])
        
        p_time, dep = -1, 0 # tm -1 initiator
        
        leftover = 0 #numpy.zeros((DEPTH), dtype= 'int32')
        blen = 0 #* numpy.ones((DEPTH), dtype= 'int32')

        st_arr = sorted(lis[i], key=lambda x: x[0] if x[0] != 0 or x[1] != 0 and x[3] != 0 else sys.maxsize)
        print(st_arr, "--st_arr")
        st_arr = numpy.array(st_arr)
        # print(st_arr, "")
        for j in range(le):
            
            timest, note, velo, dura = (int(v) for v in st_arr[j])

            if dep == DEPTH: 
                dep = 0
            if p_time == timest and dep == 0: 
                continue
            elif p_time == timest:
                # if leftover[dep - 1] == 0: blen[dep - 1] += dura
                # else : blen += leftover[dep - 1]
                blen -= dura
                pass
            elif p_time != timest:
                dep = 0
                
            if blen + dura >= 128:
                leftover = abs(128 - blen - dura)
            else : leftover = 0

            if blen > 127: 
                roll[i].append(broll.tolist())
                broll = numpy.zeros((128, DEPTH))  #total 128 sFlat Time instances, 4 MEASURES ------------> list syntax : [[0 for _ in range(DEPTH)] for _ in range(128)]
                blen = 0 
                dep = 0
                 
                #Dealing with leftover ... 
                # if leftover.any():
                #     for ind, l in enumerate(leftover):
                #         broll[blen - l : blen, ind] = note
                #         leftover[ind] = 0
                #     blen -= dura
            
            # print(timest, note, velo, dura)
            
            broll[blen : blen + dura - leftover - 1, dep] = note
            # broll[blen + dura - leftover - 1, dep] = 0
            blen += dura
            dep += 1
            p_time = timest
            broll    
        # print(broll.tolist(), broll.shape, "--broll")
        # print(roll, "---roll")
    roll[i].append(broll.tolist())
    print("------------------", blen , broll.tolist())

    # print(roll, "---roll")
    
    
    ar_roll = numpy.array(roll)
    le = max([len(a) for a in ar_roll])
    ar_roll = numpy.zeros((ar_roll.shape[0], le, 128, DEPTH))
    for i in range(ar_roll.shape[0]):
        ln = len(roll[i])
        for j in range(ln):
            ar_roll[i, j] = roll[i][j]
    print(ar_roll, ar_roll.shape, "--roll shape") 
    # roll = numpy.roll(ar_roll, 128 - blen, axis = 1)   
    return roll


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


# def sFlat_to_fFlat(sflat):
    # print(sflat.shape, "--sflat")

def sflat_to_csflat(sflat):
    scroll = numpy.zeros(sflat.shape[:-1] + (1, ))


    for i in range(sflat.shape[0]):
        for j in range(sflat.shape[1]):
            notes = sorted(sflat[i, j]) #sorting the notes in ascending order
            diff = numpy.diff(notes)
            if j < 5:
                print(diff, "--diff")

def merge_rolls(arr1, arr2, axis = 0):
    return numpy.concatenate((arr1, arr2), axis = axis)


def sync_tarray(t_arr1, t_arr2):
    mn1 = min(t_arr1[0, :, 3])
    mn2 = min(t_arr2[0, : , 3])

    print(t_arr1.shape, t_arr2.shape, "-- tarr1, tarr2")

    print(mn1, mn2, "--mn1, mn2")

    le_mn = min(t_arr1.shape[1], t_arr2.shape[1])

    mn = min([mn1, mn2]) #loop ticker

    ctarr1, ctarr2=  [], []

    itr1, itr2 = 0, 0

    tm = 0

    while itr1 < t_arr1.shape[1] and itr2 < t_arr2.shape[1]: #Time steps
      
        if t_arr1[0, itr1, 0] == t_arr2[0, itr2, 0]:

            ctarr1.append(t_arr1[0, itr1].tolist())
            ctarr2.append(t_arr2[0, itr2].tolist())
            itr1 += 1
            itr2 += 1
        elif t_arr1[0, itr1, 0] > t_arr2[0, itr2, 0]:

            ctarr2.append(t_arr2[0, itr2].tolist())
            ctarr1.append([t_arr2[0, itr2, 0], 0 ,0 ,0])
            itr2 += 1
        elif t_arr1[0, itr1, 0] < t_arr2[0, itr2, 0]:

            ctarr2.append([t_arr1[0, itr1, 0], 0 ,0 ,0])
            ctarr1.append(t_arr1[0, itr1].tolist())
            itr1 += 1

    ctarr1 = numpy.array(ctarr1)
    ctarr2 = numpy.array(ctarr2)

    return numpy.reshape(ctarr1, (1, ) + ctarr1.shape), numpy.reshape(ctarr2, (1, ) + ctarr2.shape)

def ip_patterns_to_tarray(ip_pat, delt = 8):
    tarr = numpy.zeros((1, len(ip_pat[0]), 4))
    tm = 0
    for i, v in enumerate(ip_pat[0]):
        tarr[0][i] = [tm, ip_pat[0][i], 80, delt]
        tm += delt
    return tarr

def ip_patterns_to_sFlat(ip_pat, DEPTH = 1):
    tarr = ip_patterns_to_tarray(ip_pat)
    return dataset.ns_tarray_to_sFlat(t_arr=tarr, DEPTH= DEPTH)

def ip_patterns_to_octave(ip_pat):
    sflat = ip_patterns_to_sFlat(ip_pat)
    return enc_deco.sFlat_to_octave(sflat)

def ip_patterns_to_inp_shape(ip_pat, shape = (1, 32, 1, 2, 16)):
    enc = ip_patterns_to_octave(ip_pat)
    return numpy.reshape(enc,  enc.shape)


def ns_tarray_to_sFlatroll(tarray, quanta = 1, depth = 3):
    #quata : 1 32ndth note
    tarray = numpy.array(tarray, 'int32')
    # print(tarray)
    size = tarray[:][:][0].max() + 3
    print(size, "--size")
    sflatroll = numpy.zeros((len(tarray), size, depth))
    dp = numpy.zeros(size, dtype='int32')
    for i, t in enumerate(tarray):
        prev_tm = 0
        dp.fill(0)
        for j, v in enumerate(t):
            # print(v[0])
            if dp[tarray[i][j][0]] >= depth: continue
            sflatroll[i, tarray[i][j][0] : tarray[i][j][0] + tarray[i][j][3] , dp[tarray[i][j][0]]] = tarray[i][j][1]
            dp[tarray[i][j][0] : tarray[i][j][0] + tarray[i][j][3]]  += 1

    return sflatroll

def sFlatroll_to_ns_tarray(roll):
    lis = []
    depth =  roll.shape[2]
    prevcount = numpy.zeros(depth)
    prev = numpy.zeros(depth)
    i = 0
    def check_insertion(row):
        if lis != []:
            prevrow = lis[-1]
            if prevrow[0] == row[0] and row[3] > prevrow[3]:
                lis.pop(-1)
                lis.append(row)
                lis.append(prevrow)
            else : lis.append(row)
        else : lis.append(row)
    
    def to_insert(j):
        elecount = prevcount[j] // prev[j] #int division 
        elestart = i - elecount
        row = [elestart, prev[j], 80, elecount]
        check_insertion(row) 
        


    for ti in range(roll.shape[0]):
        for i in range(roll.shape[1]):
            ele = 0
            for j in range(depth):
                ele = roll[ti, i, j] #the element
                if ele != prev[j] and prev[j] != 0:
                    to_insert(j)

                    prev[j] = ele
                    prevcount[j] = ele
                else:
                    if prev[j] == 0: prev[j] = ele
                    prevcount[j] += ele
        
        for j in range(depth):
            if prev[j] == 0: continue
            to_insert(j)
        
        print(lis)


# def sFlatroll_to_ns_tarray(roll):
#     tarr = []
#     depth = roll.shape[2]
#     for t, tr in enumerate(roll):
#         tarr.append([])
#         for j in range(depth):
#             cnt = 0 
#             for k, tm in enumerate(tr):
#                 if tarr[t] == []:
#                     tarr[t].append([cnt, tm[j], 80, 1])
#                     cnt += 1
#                 if tm[j] == 0:
#                     cnt += 1 
#                     continue
#                 if tarr[t][-1][1] == tm[j]: #prev note same
#                     tarr[t][-1][0] += 1
#                     tarr[t][-1][3] += 1
#                     cnt += 1
#                 else :
#                     tarr[t].append([cnt, tm[j], 80, 1])
#                     cnt += 1
#     le = (len(arr) for arr in tarr)
#     print(le)
#     size  = max(len(arr) for arr in tarr)
#     tarr1 = numpy.zeros((roll.shape[0], size, 4), dtype = 'int32')
#     for t in range(roll.shape[0]):
#         tarr1[t, :len(tarr[t])] = tarr[t]
#     print(tarr1, tarr1.shape, " -- tarr")
                
                






