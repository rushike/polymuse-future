import numpy, copy


"""
All utility functionalites majorly related to dataset in and out , i.e. preprocessing step

"""

## ################################################# # #################################################
#  This includes some functions from mutils, a program from rmidi                                   ####
#                                                                                                   ####
# # ################################################# # #################################################

import math, re
import hashlib as ha
import os, glob

hash_ =  ha.md5()

WRAP_DATA = 0x7f
WRAP_BITS = 7

def hex2(val):
    return "0x{:02x}".format(val)

@DeprecationWarning
def hexn(val, n):
    return "0x{:0{}x}".format(val, n)

def ch_event_id(id, no) :
    return ((id & 0xf) << 4) | (no & 0xf)

def channel(evt_id):
    return (evt_id >> 4) & 0xf
    
def numin(num, startb, off):#off length of bits that want
    return (num >> startb) & ((1 << off) - 1)

def meta_event_type(typ):
    return typ & 0xff

def invert(num, bits = 3, twos = False):
    # if bits == 3: return ~num & 7
    twos = 1 if twos else 0
    MASK = (1 << bits) - 1
    return ((~num & MASK) + twos)  & MASK  

def magnitude(num, bits):
    """Only use for signed integer
    
    Arguments:
        num {number} -- mag(num)
        bits {number} -- bits used to encode
    """
    MASK = (1 << bits - 1) - 1
    if num > MASK: return ((~num & MASK) + 1)
    return num & MASK
    

def to_var_length(k):
    if k > 127:
        leng = length(k) // WRAP_BITS + 1
        var = bytearray(leng)
        var[-1] = k & WRAP_DATA
        k >>= WRAP_BITS
        for i in range(leng - 2, -1, -1):
            var[i] = (k & WRAP_DATA) | (1 << WRAP_BITS )
            k >>= WRAP_BITS
        return var
    else:
        return bytearray((k,))     

def to_fix_length(k, leng, bits):
    fix = bytearray(leng)
    wrapper = (1 << bits) - 1
    if k > 255 and k > -1:
        fix[-1] = k & wrapper
        k >>= bits
        for i in range(leng - 2, -1, -1):
            # fix[i] = (k & wrapper) + wrapper + 1
            fix[i] = k & wrapper
            k >>= bits
    elif k > -1 :
        fix[-1] = k & 0xff
    return fix


def vartoint(varray:bytearray()):
    return toint(varray, 7)

def toint(a : bytearray(), bits = 8, mode = 'BG'):
    WRAPPER = (1 << bits) - 1
    num, itr, ind, s, le = 0x00, 0, 0, 1, len(a)
    if mode == 'LL': #set loop from end for little indian  
        s, ind= -1, le - 1
    while itr < le and ind < le:
        num = (num << bits) | (a[ind] & WRAPPER)
        ind += s
        itr += 1
    return num


def up(n_b, base = 2):
    return base ** int(math.log2(n_b) / math.log2(base) + 1)
    
def split(n, t, block_size = None): #Splits 'n' integer in t integer of bits bit(n)/t
    n_b = length(n)
    sbit = up(n_b) // t if not block_size else block_size
    WRAPPER = 2 ** sbit - 1
    k = n
    li = []
    while k:
        li.append(k & WRAPPER)
        k >>= sbit
    le = t - len(li)
    for _ in range(le):
        li.append(0)
    li.reverse()
    return li


def merge(a, b, *nums, block_size = 16):#Merge the numbers in order
    WRAPPER = (1 << block_size) - 1
    num = ((a & WRAPPER) << block_size) + (b & WRAPPER)
    for v in nums:
        num = (num << block_size) + (v & WRAPPER)
    return num 

def length(k):
    """Finds length of k in bits
    
    Arguments:
        k {int} -- Integer number
    """
    return int(math.log2(k) + 1)

def match(whole:bytearray, pattern= bytearray):
    return re.search(pattern, whole)

def find_location(text, listt):
    try:
        return next((i, j) 
            for i, t in enumerate(listt)
            for j, v in enumerate(t)
            if v == text)
    except StopIteration:
        return None

# @param num byte array to be converted
# @param type type : big-endian or small-endian
# @param group no. of bytes in group
# @param length length of line(in bytes)
# @return
def hexstr(bnum: bytearray, leng = 0, group = 0, numlen = 2, ftype = 0):
    if ftype == 1: bnum = bnum.reverse()
    if group == 0: group = len(bnum)
    if leng == 0: leng = len(bnum)    
    st = ''
    for i in range(len(bnum)):
        x = hex(bnum[i])[2:]
        if len(x) != numlen: x = "0" * (numlen - len(x)) + x[:2]
        st += ('0x' + x + ' ')
        if i % group == 0: st += ' '
        if (i + 1) % leng == 0: st += '\n'
    return st + '\n' if st == '' else st

def dtime(delta_time, time_div):
    if not delta_time: return 0
    return int((time_div * 4) // delta_time)


def file_hash(f, hexst = False):
    with open(f, "rb") as fi:
        cont = fi.read()
        return ha.md5(cont).hexdigest()
    return hash_.hexdigest()

def midi_to_note(noteval):
    if not numpy.isscalar(noteval): return [midi_to_note(v) for v in noteval]

    if not 0 <= noteval < 128: raise ValueError('Noteval not in range : {}'.format(noteval))
    
    mod = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    
    note = mod[noteval % 12]
    octave = noteval // 12 - 1
    return '{}{}'.format(note, octave)

@DeprecationWarning
def note_to_midi(note): #Not implemented
    val_notes = {'c' : 0, 'c#' : 1, 'd' : 2, 'd#': 3, 'e': 4, 'f': 5, 'f#': 6, 'g': 7, 'g#': 8, 'a': 9, 'a#': 10, 'b': 11}
    if not numpy.isscalar(note): return [note_to_midi(v) for v in note]
    
    if note[:2].lower() not in val_notes: raise ValueError('Invalid Note String {}'.format(note))

    note, octave = note[:2], note[2:]


def dictn(ndlist):
    return {l[0] : l[1:]  for l in ndlist}

def get_all_midis_gen(folder_path, generator = False, maxx = 10):
    # os.chdir(folder_path)
    f1 = []
    mx = 0
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
#             print(os.path.join(root, name))
            if generator and mx >= maxx: return f1
            p = os.path.join(root, name)
            if p.endswith('.mid') or p.endswith('.midi'):
                if generator: yield p
                f1.append(p)
                mx += 1
    return f1

def get_all_midis(folder_path, generator = False, maxx = 10):
    # os.chdir(folder_path)
    f1 = []
    mx = 0
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
#             print(os.path.join(root, name))
            if mx >= maxx: return f1
            p = os.path.join(root, name)
            if p.endswith('.mid') or p.endswith('.midi'):
                f1.append(p)
                mx += 1
    return f1

def get_all_files(folder_path, typ = '', maxx = 1):
    f1 = []
    mx = 0
    typ = None if typ == '' else typ if typ.startswith('.') else '.' + typ
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
#             print(os.path.join(root, name))
            if mx >= maxx: return f1
            p = os.path.join(root, name)
            if typ:
                if typ and p.endswith(typ) or p.endswith(typ):
                    f1.append(p)
                    mx += 1
            else :
                f1.append(p)
                mx += 1
    return f1


def sample(preds, temperature= 1.0, choice_random = False):
    # helper function to sample an index from a probability array
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    if choice_random: return numpy.random.choice(probas.shape[0], p = probas)
    return numpy.argmax(probas)


# def create_path(path, HOME = os.getcwd()):
#     path = path.split('/')
#     if path[-1] != '': path.pop()
#     home = HOME
#     for d in path:
#         if not os.path.exists(d): 
#             os.mkdir(home)
#             os.chdir(d)
#     os.chdir(HOME)
        

def quater_note_to_millis(tempo):
    return tempo / 60000

def nth_note(duration, tempo):
    th2 = quater_note_to_millis(tempo) / 8
    wh_note = th2 * 32
    to_note = duration / th2
    rt = int(numpy.round(to_note))
    # rt_str = '{0:05b}'.format(rt)
    fin = 32 / rt if rt != 0 else 0
    return fin



def find_in_nested_dict(dictn, value, depth = 0):
    """ return dict containg the value in nested dict
    
    Arguments:
        dictn {dict} -- nested dictionart
        value {int} -- value to search
    
    Keyword Arguments:
        depth {int} -- depth value in nested dictionary (default: {0})
    
    Raises:
        ValueError: if value is not present in dict
    
    Returns:
        [dict] -- dict containing the value
    """
    for k, v in dictn:
        if v == value:
            return dictn
        elif hasattr(v, 'items'): # indicates if dictionary
            return find_in_nested_dict(v, value, depth - 1)
        else : raise ValueError("Value not found in the nested dictionary")

def reduce_dim(n, factor = 8):
    T = len(n) // factor
    res = numpy.zeros(T)
    
    for i in range(T):
        res[i] = sum(n[i : i + factor]) / factor

    return res

# #################################################
# # #################################################
# # #################################################

def to_numpy_array_from_3D_list(listn, shape = [3, 1000, 5], depth = 1):
    # if depth == 0:
    # le = len(listn)
    print(shape)
    res = numpy.zeros(shape)
    for i in range(shape[0]):
        le = len(listn[i])
        for j in range(shape[1]):
            if j >= le: break
            for k in range(shape[2]):    
                res[i, j , k] = listn[i][j][k]

    return res


def note_length(note_len, resolution = 32):
    if note_len == 0: return 0 
    return resolution // note_len

def translate_01_axis(roll, sw_axis = [0, 1]): #translate 1 axis to zeros one, (n, m, ...)   <--->  (m, n, ...)
    # ax_1, ax_2 = sw_axis[0], sw_axis[1]
    # shp = list(roll.shape)
    # shp[ax_1], shp[ax_2] = roll.shape[ax_2], roll.shape[ax_1]
    # n_roll = numpy.zeros(shp)
    return numpy.swapaxes(roll, sw_axis[0], sw_axis[1])
    # for i in range(shp[ax_1]):
    #     for j in range(shp[ax_2]):
    #         n_roll[i][j] = roll[j][i]
    #     pass 
    # pass
    # return n_roll

# def to_bin(arr, spread = 8):
#     mx_axis = len(arr.shape)
#     arrl = []
#     for i in range(arr.shape[0]):
#         if mx_axis >= 1:
#             if mx_axis == 1: arrl[i] = bin_arr(arr[i], spread)
#         else : continue
#         arrl[i].append([])
#         for j in range(arr.shape[1]):
#             if mx_axis >= 2: 
#                 if mx_axis == 2: arrl[i][j] = bin_arr(arr[i, j], spread)
#             else : continue
#             arrl[i][j].append([])
#             for k in range(arr.shape[2]):
#                 if mx_axis >= 3: 
#                     if mx_axis == 3: arrl[i][j][k] = bin_arr(arr[i, j, k], spread)
#                 else : continue
#                 arrl[i][j][k].append([])
#                 for l in range(arr.shape[3]):
#                     if mx_axis >= 4: 
#                         if mx_axis == 4: arrl[i][j][k][l] = bin_arr(arr[i,  j, k, l], spread)
#                     else : continue
#                     arrl[i][j][k][l].append([])
#                     for m in range(arr.shape[4]):
#                         if mx_axis >= 5:
#                             if mx_axis == 5: arrl[i][j][k][l][m] = bin_arr(arr[i, j, k, l, m], spread)
#                         else : continue
#                         arrl[i][j][k][l][m].append([])
#                         for n in range(arr.shape[5]):
#                             if mx_axis >= 6: 
#                                 if mx_axis == 6: arrl[i][j][k][l][m][n] = bin_arr(arr[i, j, k, l, m, n], spread)
#                             else : continue
#                             # arrl[i][j][k][l][m][n].

#     return numpy.array(arrl)

def to_3D_bin(arr, spread):
    res_shape = list(arr.shape)
    arr = numpy.array(arr, dtype = 'int32')
    res_shape.extend([spread])
    print(res_shape)
    res = numpy.zeros(res_shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                # print("num to pass : ", arr[i][j][k])
                res[i, j, k] = bin_arr(arr[i, j, k], spread)

    return res
    

def rev_bin_3D(arr, spread):
    # arr = numpy.array(arr, dtype = 'int32')
    # res_shape.extend([spread])
    # print("arr -> jklu :", arr.shape[:-1])
    res = numpy.zeros(arr.shape[:-1])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                # print("num to pass : ", arr[i][j][k])
                res[i, j, k] = rev_bin_arr(arr[i, j, k], spread)
    # print(res.shape)
    return res





def rev_bin_arr(ar, spread):
    num = 1 << (spread - 1)
    # print(num)
    res = 0
    for i in range(spread):
        res += (ar[i] * num)
        num //= 2
    return res

def bin_arr(num, spread):
    stl = bin(num)
    # print("Binart val: ", stl)
    st = stl[2:]
    le = len(st)
    k = spread - le
    i = 0
    bin_ar = numpy.zeros(spread, dtype = 'int32')
    for i in range(le):
        bin_ar[i + k] = int(st[i])
    # print(bin_ar)
    return bin_ar


def one_hot(ar, classes):
    ar = numpy.array(ar)
    shape = ar.shape + (classes, )
    res = numpy.zeros(shape)
    for i in range(ar.shape[0]):
        for j in range(ar.shape[1]):
            res[i, j] = to_categorical(ar[i, j], classes)
    return res    

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Arguments:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The classes axis is placed
        last.
    """
    y = numpy.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = numpy.max(y) + 1
    n = y.shape[0]
    categorical = numpy.zeros((n, num_classes), dtype=dtype)
    categorical[numpy.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = numpy.reshape(categorical, output_shape)
    return categorical

def rev_one_hot(ar):
    ar = numpy.array(ar)
    shape = ar.shape[:-1] + (1,)
    print(shape)
    res = numpy.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            index = numpy.argmax(ar[i, j])
            res[i, j, 0] = index if numpy.isscalar(index) else index[0]
    return res

def trim_axis2(arr):
    N = arr.shape[1]
    for j in range(N - 1, -1, -1):
        if arr[0, j].any(): break
    return arr[:, :j + 1]


def thresholding(arr, thr):
    arr[arr >= thr] = 1
    arr[arr < thr] = 0
    return arr

def arg_max(ar):
    # return thresholding(ar, numpy.max(ar))
    m = numpy.argmax(ar)
    # print(m)
    ar[True] = 0
    if numpy.isscalar(m) : ar[m] = 1
    else : ar[m[0]] = 1
    return ar

def tm(ar):
    return numpy.argmax(ar)

def arg_octave_max(octn):
    mx0 = numpy.argmax(octn[0])
    mx1 = numpy.argmax(octn[1])
    octn[0, mx0] = 1
    octn[0, octn[0] != 1] = 0
    octn[1, mx1] = 1
    octn[1, octn[1] != 1] = 0
    return octn

def arg_oct_max(octn, freqn, size= 16):
    res = numpy.zeros((2, size))
    res[0, octn] = 1
    res[1, freqn] = 1
    return res