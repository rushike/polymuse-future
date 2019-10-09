import numpy

"""
Constants : This files includes the basic constant used with polymuse from instrument codes, to inverse of it, to scale patterns etc ... 

"""


def instrument_program_code(names):
    codes = numpy.zeros(len(names), dtype='int32')
    for i, nm in enumerate(names):
        codes[i] = inst_codes[nm.lower()]
    return codes


rev_inst_codes = {0: 'Acoustic Grand Piano', 1: 'Bright Acoustic Piano', 2: 'Electric Grand Piano', 3: 'Honky-tonk Piano', 4: 'Electric Piano 1', 5: 'Electric Piano 2', 6: 'Harpsichord', 7: 'Clavi', 8: 'Celesta', 9: 'Glockenspiel', 10: 'Music Box', 11: 'Vibraphone', 12: 'Marimba', 13: 'Xylophone', 14: 'Tubular Bells', 15: 'Dulcimer', 16: 'Drawbar Organ', 17: 'Percussive Organ', 18: 'Rock Organ', 19: 'Church Organ', 20: 'Reed Organ', 21: 'Accordion', 22: 'Harmonica', 23: 'Tango Accordion', 24: 'Acoustic Guitar (nylon)', 25: 'Acoustic Guitar (steel)', 26: 'Electric Guitar (jazz)', 27: 'Electric Guitar (clean)', 28: 'Electric Guitar (muted)', 29: 'Overdriven Guitar', 30: 'Distortion Guitar', 31: 'Guitar harmonics', 32: 'Acoustic Bass', 33: 'Electric Bass (finger)', 34: 'Electric Bass (pick)', 35: 'Fretless Bass', 36: 'Slap Bass 1', 37: 'Slap Bass 2', 38: 'Synth Bass 1', 39: 'Synth Bass 2', 40: 'Violin', 41: 'Viola', 42: 'Cello', 43: 'Contrabass', 44: 'Tremolo Strings', 45: 'Pizzicato Strings', 46: 'Orchestral Harp', 47: 'Timpani', 48: 'String Ensemble 1', 49: 'String Ensemble 2', 50: 'SynthStrings 1', 51: 'SynthStrings 2', 52: 'Choir Aahs', 53: 'Voice Oohs', 54: 'Synth Voice', 55: 'Orchestra Hit', 56: 'Trumpet', 57: 'Trombone', 58: 'Tuba', 59: 'Muted Trumpet', 60: 'French Horn', 61: 'Brass Section', 62: 'SynthBrass 1', 63: 'SynthBrass 2', 64: 'Soprano Sax', 65: 'Alto Sax', 66: 'Tenor Sax', 67: 'Baritone Sax', 68: 'Oboe', 69: 'English Horn', 70: 'Bassoon', 71: 'Clarinet', 72: 'Piccolo', 73: 'Flute', 74: 'Recorder', 75: 'Pan Flute', 76: 'Blown Bottle', 77: 'Shakuhachi', 78: 'Whistle', 79: 'Ocarina', 80: 'Lead 1 (square)', 81: 'Lead 2 (sawtooth)', 82: 'Lead 3 (calliope)', 83: 'Lead 4 (chiff)', 84: 'Lead 5 (charang)', 85: 'Lead 6 (voice)', 86: 'Lead 7 (fifths)', 87: 'Lead 8 (bass + lead)', 88: 'Pad 1 (new age)', 89: 'Pad 2 (warm)', 90: 'Pad 3 (polysynth)', 91: 'Pad 4 (choir)', 92: 'Pad 5 (bowed)', 93: 'Pad 6 (metallic)', 94: 'Pad 7 (halo)', 95: 'Pad 8 (sweep)', 96: 'FX 1 (rain)', 97: 'FX 2 (soundtrack)', 98: 'FX 3 (crystal)', 99: 'FX 4 (atmosphere)', 100: 'FX 5 (brightness)', 101: 'FX 6 (goblins)', 102: 'FX 7 (echoes)', 103: 'FX 8 (sci-fi)', 104: 'Sitar', 105: 'Banjo', 106: 'Shamisen', 107: 'Koto', 108: 'Kalimba', 109: 'Bag pipe', 110: 'Fiddle', 111: 'Shanai', 112: 'Tinkle Bell', 113: 'Agogo', 114: 'Steel Drums', 115: 'Woodblock', 116: 'Taiko Drum', 117: 'Melodic Tom', 118: 'Synth Drum', 119: 'Reverse Cymbal', 120: 'Guitar Fret Noise', 121: 'Breath Noise', 122: 'Seashore', 123: 'Bird Tweet', 124: 'Telephone Ring', 125: 'Helicopter', 126: 'Applause', 127: 'Gunshot'}


inst_codes = {
                'piano' : 0, 'acoustic grand piano': 0, 'bright acoustic piano': 1, 'electric grand piano': 2, 'honky-tonk piano': 3, 'electric piano 1': 4, 'electric piano 2': 5, 'harpsichord': 6, 'clavi': 7, 'celesta': 8,
                'glockenspiel': 9, 'music box': 10, 'vibraphone': 11, 'marimba': 12, 'xylophone': 13, 'tubular bells': 14, 'dulcimer': 15, 'drawbar organ': 16, 
                'percussive organ': 17, 'rock organ': 18, 'church organ': 19, 'reed organ': 20, 'accordion': 21, 'harmonica': 22, 'tango accordion': 23, 'acoustic guitar (nylon)': 24, 
                'guitar' : 25, 'acoustic guitar (steel)': 25, 'electric guitar (jazz)': 26, 'electric guitar (clean)': 27, 'electric guitar (muted)': 28, 'overdriven guitar': 29, 'distortion guitar': 30, 'guitar harmonics': 31, 'acoustic bass': 32, 
                'electric bass (finger)': 33, 'electric bass (pick)': 34, 'fretless bass': 35, 'slap bass 1': 36, 'slap bass 2': 37, 'synth bass 1': 38, 'synth bass 2': 39, 'violin': 40, 
                'viola': 41, 'cello': 42, 'contrabass': 43, 'tremolo strings': 44, 'pizzicato strings': 45, 'orchestral harp': 46, 'timpani': 47, 'string ensemble 1': 48, 
                'string ensemble 2': 49, 'synthstrings 1': 50, 'synthstrings 2': 51, 'choir aahs': 52, 'voice oohs': 53, 'synth voice': 54, 'orchestra hit': 55, 'trumpet': 56, 
                'trombone': 57, 'tuba': 58, 'muted trumpet': 59, 'french horn': 60, 'brass section': 61, 'synthbrass 1': 62, 'synthbrass 2': 63, 'soprano sax': 64, 
                'alto sax': 65, 'tenor sax': 66, 'baritone sax': 67, 'oboe': 68, 'english horn': 69, 'bassoon': 70, 'clarinet': 71, 'piccolo': 72, 
                'flute': 73, 'recorder': 74, 'pan flute': 75, 'blown bottle': 76, 'shakuhachi': 77, 'whistle': 78, 'ocarina': 79, 'lead 1 (square)': 80, 
                'lead 2 (sawtooth)': 81, 'lead 3 (calliope)': 82, 'lead 4 (chiff)': 83, 'lead 5 (charang)': 84, 'lead 6 (voice)': 85, 'lead 7 (fifths)': 86, 'lead 8 (bass + lead)': 87, 'pad 1 (new age)': 88, 
                'pad 2 (warm)': 89, 'pad 3 (polysynth)': 90, 'pad 4 (choir)': 91, 'pad 5 (bowed)': 92, 'pad 6 (metallic)': 93, 'pad 7 (halo)': 94, 'pad 8 (sweep)': 95, 'fx 1 (rain)': 96, 
                'fx 2 (soundtrack)': 97, 'fx 3 (crystal)': 98, 'fx 4 (atmosphere)': 99, 'fx 5 (brightness)': 100, 'fx 6 (goblins)': 101, 'fx 7 (echoes)': 102, 'fx 8 (sci-fi)': 103, 'sitar': 104, 
                'banjo': 105, 'shamisen': 106, 'koto': 107, 'kalimba': 108, 'bag pipe': 109, 'fiddle': 110, 'shanai': 111, 'tinkle bell': 112, 
                'agogo': 113, 'steel drums': 114, 'woodblock': 115, 'taiko drum': 116, 'melodic tom': 117, 'synth drum': 118, 'reverse cymbal': 119, 'guitar fret noise': 120, 
                'breath noise': 121, 'seashore': 122, 'bird tweet': 123, 'telephone ring': 124, 'helicopter': 125, 'applause': 126, 'gunshot': 127
            }


chord_prog = {
                'major-3' : [4, 5, 0, 0],
                'minor-3' : [3, 5, 0, 0]
            }

chord_prog_to_codes = { 
                'major-3' : 0,
                'minor-3' : 1,
            }
codes_to_chord_prog = { 
                0 : 'major-3',
                1 : 'minor-3',
            }

scale_patterns = {
                'major' : [0, 2, 2, 1, 2, 2, 2, 1],
                'minor' : [0, 2, 1, 2, 2, 2, 1, 2],
            }

scale_patterns_cum = {
                'major' : [0, 2, 4, 5, 7, 9, 11, 12],
                'minor' : [0, 2, 3, 5, 7, 8, 10, 12],
            }


scale_names = {
        'major' : ['C', 'C#', 'D ', 'D#', 'E ', 'F ', 'F#', 'G ', 'G#', 'A ', 'A#', 'B '],
        'minor' : ['Cm', 'C#m', 'Dm ', 'D#m', 'Em ', 'Fm ', 'F#m', 'Gm ', 'g#m', 'Am ', 'A#m', 'Bm '],       
    }

lead_track = [0, 1, 2, 4, 6, 7, 8,  13,  15,  33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,  70,  72]


model_dict = {
        'lead' : 'lead', 'piano' : 'lead', 'first' : 'lead',
        'chorus' : 'chorus', 'second' : 'chorus',
        'drum' : 'drum', 'third' : 'drum'
    }

depths_of_3tracks = {
                    0 : 1, 1: 3, 2: 2, 
                    'lead' : 1, 'chorus' : 3, 'drum': 2 
                    }