

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
3 files will generated and stored in *dir* strucure :
h5_models
├── chorus                                                                                                              
│   └── stateless                                                                                                       
│       └── wlvv.h5                                                                                                     
├── drum                                                                                                                
│   └── stateless                                                                                                       
│       └── vyvh.h5                                                                                                     
└── lead                                                                                                                    
    └── stateless                                                                                                               
        └── vyvh.h5