import matplotlib.pyplot as plt
import numpy, json

def draw_json_loss_acc(j_fn, j_ft): #draw note and time
    with open(j_fn, 'r') as j_file:
        j_strn = j_file
        dictn = json.load(j_strn)

    xn_loss = dictn['loss']
    xn_val_loss = dictn['val_loss']

    xn_acc = dictn['acc']
    xn_val_acc = dictn['val_acc']

    with open(j_ft, 'r') as j_file:
        j_strt = j_file
        dictn = json.load(j_strt)

    xt_loss = dictn['loss']
    xt_val_loss = dictn['val_loss']

    xt_acc = dictn['acc']
    xt_val_acc = dictn['val_acc']

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(2, 2, 1)
    x = numpy.arange(len(xn_loss))
    ax.plot(x, xn_loss, label = "Loss vs Epochs")
    ax.plot(x, xn_val_loss, label = "Val Loss vs Epochs")    
    ax.title.set_text("Note Loss")
    # ax.xlabel("Epochs")
    # ax.ylabel("Loss")
    ax.legend()

    ax = fig.add_subplot(2, 2, 2)
    x = numpy.arange(len(xn_loss))
    ax.plot(x, xn_acc, label = "Acc vs Epochs")
    ax.plot(x, xn_val_acc, label = "Val Acc vs Epochs")
    ax.title.set_text("Note Acc")
    # ax.xlabel("Epochs")
    # ax.ylabel("Loss")
    ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    x = numpy.arange(len(xt_loss))
    ax.plot(x, xt_loss, label = "Loss vs Epochs")
    ax.plot(x, xt_val_loss, label = "Val Loss vs Epochs") 
    ax.title.set_text("Time Loss")   
    # ax.xlabel("Epochs")
    # ax.ylabel("Loss")
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    x = numpy.arange(len(xt_loss))
    ax.plot(x, xt_acc, label = "Acc vs Epochs")
    ax.plot(x, xt_val_acc, label = "Val Acc vs Epochs")
    ax.title.set_text("Time Acc")
    # ax.xlabel("Epochs")
    # ax.ylabel("Loss")
    ax.legend()
    

    plt.show()

def draw_json_oct_loss_acc(j_fn, j_ft): #draw note and time
    with open(j_fn, 'r') as j_file:
        j_strn = j_file
        dictn = json.load(j_strn)

    xn_loss = dictn['loss']
    xn_val_loss = dictn['val_loss']

    xn_acc = dictn['acc']
    xn_val_acc = dictn['val_acc']

    
    xn_ocloss = dictn['octave_loss']
    xn_val_ocloss = dictn['val_octave_loss']

    with open(j_ft, 'r') as j_file:
        j_strt = j_file
        dictn = json.load(j_strt)

    xt_loss = dictn['loss']
    xt_val_loss = dictn['val_loss']

    xt_acc = dictn['acc']
    xt_val_acc = dictn['val_acc']

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(2, 2, 1)
    x = numpy.arange(len(xn_loss))
    ax.plot(x, xn_loss, label = "Loss vs Epochs")
    ax.plot(x, xn_val_loss, label = "Val Loss vs Epochs")    
    ax.plot(x, xn_ocloss, label = "Oct Loss vs Epochs")    
    ax.plot(x, xn_val_ocloss, label = "Val Oct Loss vs Epochs")    
    ax.title.set_text("Note Loss")
    # ax.xlabel("Epochs")
    # ax.ylabel("Loss")
    ax.legend()

    ax = fig.add_subplot(2, 2, 2)
    x = numpy.arange(len(xn_loss))
    ax.plot(x, xn_acc, label = "Acc vs Epochs")
    ax.plot(x, xn_val_acc, label = "Val Acc vs Epochs")
    ax.title.set_text("Note Acc")
    # ax.xlabel("Epochs")
    # ax.ylabel("Loss")
    ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    x = numpy.arange(len(xt_loss))
    ax.plot(x, xt_loss, label = "Loss vs Epochs")
    ax.plot(x, xt_val_loss, label = "Val Loss vs Epochs") 
    ax.title.set_text("Time Loss")   
    # ax.xlabel("Epochs")
    # ax.ylabel("Loss")
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    x = numpy.arange(len(xt_loss))
    ax.plot(x, xt_acc, label = "Acc vs Epochs")
    ax.plot(x, xt_val_acc, label = "Val Acc vs Epochs")
    ax.title.set_text("Time Acc")
    # ax.xlabel("Epochs")
    # ax.ylabel("Loss")
    ax.legend()
    

    plt.show()


def draw_json_loss_acc_1(j_fn): #draw note and time
    with open(j_fn, 'r') as j_file:
        j_strn = j_file
        dictn = json.load(j_strn)

    xn_loss = dictn['loss']
    xn_val_loss = dictn['val_loss']

    xn_acc = dictn['acc']
    xn_val_acc = dictn['val_acc']

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(1, 2, 1)
    x = numpy.arange(len(xn_loss))
    ax.plot(x, xn_loss, label = "Loss vs Epochs")
    ax.plot(x, xn_val_loss, label = "Val Loss vs Epochs")    
    ax.title.set_text("Note Loss")
    # ax.xlabel("Epochs")
    # ax.ylabel("Loss")
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    x = numpy.arange(len(xn_loss))
    ax.plot(x, xn_acc, label = "Acc vs Epochs")
    ax.plot(x, xn_val_acc, label = "Val Acc vs Epochs")
    ax.title.set_text("Note Acc")
    # ax.xlabel("Epochs")
    # ax.ylabel("Loss")
    ax.legend()


    plt.show()
