import numpy, copy


"""
All utility functionalites majorly related to dataset in and out , i.e. preprocessing step

"""

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