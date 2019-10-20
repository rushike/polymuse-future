
from polymuse import train, player, constant


import argparse, requests, os



def parse_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-t','--train', type= str)
    ap.add_argument('-p', '--play', type= None)
    ap.add_argument('-l', '--load', type= None)

    ap.add_argument('-i', '--input_file', type= str)
    ap.add_argument('-o', '--output_file', type= str)

    ap.add_argument('-d', '--dataset_path', metavar= 'DIR', type = str)
    ap.add_argument('-m', '--maxx', metavar= 'Maximum Number of files to take for training', type = int)
    ap.add_argument('-g', '--gpu', metavar= 'if to train on gpu', type = bool, default=False)

    args = ap.parse_args()

    return args

def train_(args):
    dataset_path = args['dataset_path']
    maxx = args['maxx']
    gpu = args['gpu']
    if gpu:
        pass
        # train.train_gpu(dataset_path, maxx)
    # else : train.train(dataset_path, maxx)
    
def play_(args):
    F = args['input_file']
    O = args['output_file']
    # player.play_3_track_no_time(F, midi_fname= O)

def load():
    load_csv()

def load_csv():
    r = requests.get(constant.models_csv)
    if not os.path.isfile('.models.csv'):
        with open('.models.csv', 'wb') as f:
            f.write(r.content)

def main():
    args = parse_args()
    print(args)
    if args['train']:
        train_(args)
    if args['play']:
        play_(args)
    if args['load']:
        load()

if __name__ == '__main__':
    main()