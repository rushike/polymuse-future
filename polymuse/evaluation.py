

import numpy
import matplotlib.pyplot as plt

from polymuse import constant

def polyphony_score(tarr):
    score = numpy.zeros(3600)
    for t in tarr:
        for i, v in enumerate(t):
            if v[0] >= 3600: continue 
            score[int(v[0]): int(v[0] + v[3])] += 1

    return score[:tarr.shape[1]]


def polyphony_eval(tarr, group):
    score = polyphony_score(tarr)
    le = tarr.shape[1] // group + 1
    sc_new = numpy.zeros(le)

    for i in range(0, score.shape[0], group):
        sc_new[i // group] = sum(score[i : i + group]) / group

    return sc_new 
    

def polyphonic_index(tarr, group, vertical = 6, avg = True):
    evaln = polyphony_eval(tarr, group)
    print(evaln, '-- evaln')
    res = evaln / vertical
    return res if not avg else  res.mean()

def scale_consistency(tarr):
    scale_struct = form_scale_struct()
    sc = numpy.zeros((tarr.shape[0], scale_struct.shape[0]))
    print(scale_struct, scale_struct.shape ,", --- scale_struct")
    for i in range(tarr.shape[0]):
        total, cnt = 0, 0
        for j in range(tarr.shape[1]):
            for k in range(scale_struct.shape[0]):
                if tarr[i, j, 1] in scale_struct[k]:
                    sc[i, k] += 1
        pass
    sc /= tarr.shape[1]
    return sc   

def form_scale_struct():
    scl_struct = numpy.zeros((24, 88))
    for i in range(0, 12):
        for j in range(1, 11):
            start = j * 8
            scl_struct[i,  start : start + 8] += (numpy.array(constant.scale_patterns_cum['major']) + (12 *  j + i))
            pass
        pass
    return scl_struct

def tonal_span_abs(tarr):
    tsp = numpy.zeros(tarr.shape[1])
    for i in range(tarr.shape[0]): 
        for j in range(tarr.shape[0]):
            tsp[j] = max(tarr[i]) - min(tarr[i])
    return tsp

def tonal_span_local(tarr, group = 32):
    tsp_len = tarr.shape[1]
    tsp = numpy.zeros((tarr.shape[0], tsp_len))

    for i in range(tarr.shape[0]):
        for j in range(tarr.shape[1]):
            # MX = min([tarr.shape[1], j + 32])
            span = tarr[i, j: j + 32, 1].max() - tarr[i, j: j + 32, 1].min()
            tsp[i, j] = span
    return tsp



def view_2D(arr2D, xaxis = None, xname= 'X', yname= 'Tonal Span'):
    ts = numpy.ceil(numpy.sqrt(arr2D.shape[0]))

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    xaxis = numpy.arange(arr2D.shape[1])
    for i in range(arr2D.shape[0]):
        ax = fig.add_subplot(ts, ts, i + 1)
        ax.plot(xaxis, arr2D[i], label = yname + ' vs Track ' + str(i))
        ax.title.set_text("Track " + str(i))
        ax.legend()
    plt.show()


def view_1D(arr1D, xname= 'X', yname= 'Tonal Span'):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    x = numpy.arange(arr1D.shape[1])
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, arr1D, label = yname + ' vs Track ' + str(i))
    ax.title.set_text("Track " + str(i))
    ax.legend()
    plt.show()


