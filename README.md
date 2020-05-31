
# Shut Down, no further development

# polymuse-future
*Making the music real* 
In development phase, once completed repo will change name to polymuse

<!-- ![](https://pandao.github.io/editor.md/images/logos/editormd-logo-180x180.png)

![](https://img.shields.io/github/stars/pandao/editor.md.svg) ![](https://img.shields.io/github/forks/pandao/editor.md.svg) ![](https://img.shields.io/github/tag/pandao/editor.md.svg) ![](https://img.shields.io/github/release/pandao/editor.md.svg) ![](https://img.shields.io/github/issues/pandao/editor.md.svg) ![](https://img.shields.io/bower/v/editor.md.svg) -->

### Features
Need to discuss ....... 


**Table of Contents**
* Overview
* Components
* Links
* Installing ..
* Training(Note model)
* Loading
* Player

### Overview
This is BE project aiming to generate the musical patterns from the midi file that are the audibes to  ***ears***

### Components
Will be added soon

### Links
This to ... 
### Installing ...
This is pre complete installation, package may not run as expected

`$ pip install polymuse`

> OR

`$ pip install polymuse-future`

install the **polymuse-future** recommended


### Train
Only ***NOTE*** training available

#### Note Training
```python
from polymuse import train

F = dataset_path # It should be absolute PATH(recomended) where midi file are

train.train_gpu(F, maxx = 5) #Only if GPU is available, It uses CuDNNLstm version which performs operation on GPU
train.train(F, maxx = 5) #if GPU version do not works 
```
@dataset_path : It should be absolute PATH(recomended) where midi file are
@maxx : It is parameters that specifies maximum no of files used to training in case there are large no of files in dataset_path given

This snapshot will train the model on dataset given,
3 files will generated and stored in following *dir* strucure :
.h5_models
:...chorus                                                                                                              
:....... stateless                                                                                                       
:...........wlvv.h5                                                                                                     
:...drum                                                                                                                
:......stateless                                                                                                       
:...........vyvh.h5                                                                                                     
:...lead                                                                                                                    
:......stateless                                                                                                               
:.......... vyvh.h5

### Load Pretrain Models
Below code snapshot downloads the default model, and make above directory structure in current working directory
```python
from polymuse import loader
loader.load(mname = 'default')
```

### Load sample midis

Below code snapshot downloads the default midi and download in current directory
```python
from polymuse import loader
loader.load_midi()
```

### Testing midi : Single Song Trainig
Song : Candi Staton := Young Hearts Run Free

### Note Player
Before using the player please train the models on dataset or load pre trained models
```python
from polymuse import player
# Before this please make sure the h5_models are loaded locally

midi_file = "F:\\rushikesh\\project\\dataset\\lakh_dataset\\Kenny G" # directory where midi file will
midi_file = dutils.get_all_files(F)[0] # Midi file must be of atleast 3 tracks

player.play_3_track_no_time(midi_file, midi_fname = 'midi00')

```

The above will store midi file in current directory with file name *midi00XXX*


"""
If you need to use plot option to see the model, need to install conda, pydot , pip graphviz, conda graphviz
"""
