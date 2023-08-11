import pandas as pd
import pickle
import random

with open('C:/Users/hb/Desktop/data/NIH/train_val_list.txt', 'r') as f:
    names = f.read().splitlines() 

random.shuffle(names)

threshold = 50000

train = names[:threshold]
print(len(train))
backup = names[threshold:]
print(len(backup))

with open('C:/Users/hb/Desktop/data/NIH/train_list.txt', 'w') as f:
    for line in train:
        f.write(f"{line}\n")

with open('C:/Users/hb/Desktop/data/NIH/backup_list.txt', 'w') as f:
    for line in backup:
        f.write(f"{line}\n")
