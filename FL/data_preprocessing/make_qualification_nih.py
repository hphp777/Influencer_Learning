import pandas as pd
import pickle
import random

with open('C:/Users/hb/Desktop/data/NIH/train_val_list.txt', 'r') as f:
    names = f.read().splitlines() 

random.shuffle(names)

threshold = int(len(names) * 0.8)

train = names[:threshold]
print(len(train))
qualification = names[threshold:]
print(len(qualification))

with open('C:/Users/hb/Desktop/data/NIH/train_val_list.txt', 'w') as f:
    for line in train:
        f.write(f"{line}\n")

with open('C:/Users/hb/Desktop/data/NIH/qualification.txt', 'w') as f:
    for line in qualification:
        f.write(f"{line}\n")
