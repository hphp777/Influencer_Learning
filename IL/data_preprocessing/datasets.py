'''
Dataset Concstruction
Code based on https://github.com/FedML-AI/FedML
'''

import glob, os, sys, pdb, time
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from imageio import imread
from PIL import Image, ImageOps
import torchvision.models as models 
import torch.nn as nn
from matplotlib import pyplot as plt
import csv
import pandas as pd
import PIL.Image as pilimg
import random
import logging

import numpy as np
from numpy.core.fromnumeric import mean
import torch.utils.data as data
import torchvision.transforms as transforms
from data_preprocessing import config
import logging
import numpy as np
import torch.utils.data as data
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import DatasetFolder, ImageFolder

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class CIFAR_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        if "cifar100" in self.root:
            cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)
        else:
            cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)


        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
# Imagenet
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            imagefolder_obj = ImageFolder(self.root+'/train', self.transform)
        else:
            imagefolder_obj = ImageFolder(self.root+'/val', self.transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)
        self.target = self.samples[:,1].astype(np.int64)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)
        
class NIHTrainDataset(Dataset):
    def __init__(self,c_num, data_dir, transform = None, indices=None):
        
        self.data_dir = data_dir
        self.transform = transform
        self.df = self.get_df()
        self.make_pkl_dir(config.pkl_dir_path)
        self.the_chosen = indices
        
        self.disease_cnt = [0]*14
        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']
        
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path)):

            self.train_val_df = self.get_train_val_df()
            # pickle dump the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'wb') as handle:
                pickle.dump(self.train_val_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
        else:
            # pickle load the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'rb') as handle:
                self.train_val_df = pickle.load(handle)
        
        self.new_df = self.train_val_df.iloc[self.the_chosen, :] # this is the sampled train_val data
    
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path)):
            # pickle dump the classes list
            with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'wb') as handle:
                pickle.dump(self.all_classes, handle, protocol = pickle.HIGHEST_PROTOCOL)
                print('\n{}: dumped'.format(config.disease_classes_pkl_path))
        else:
            pass

        for i in range(len(self.new_df)):
            row = self.new_df.iloc[i, :]
            labels = str.split(row['Finding Labels'], '|')
            for lab in labels:  
                lab_idx = self.all_classes.index(lab)
                if lab_idx == 14: # No Finding
                    continue
                self.disease_cnt[lab_idx] += 1

        self.total_ds_cnt = np.array(self.disease_cnt)
        # Normalize the imbalance
        self.imbalance = 0
        difference_cnt = self.total_ds_cnt - self.total_ds_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] * difference_cnt[i]        
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] / difference_cnt.sum()
        # Calculate the level of imbalnce
        difference_cnt -= difference_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = (difference_cnt[i] * difference_cnt[i])
    
        self.imbalance = 1 / difference_cnt.sum()

        # Plot the disease distribution
        # self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation']
        # plt.figure(figsize=(8,4))
        # plt.title('Client{} Disease Distribution'.format(c_num), fontsize=20)
        # plt.bar(self.all_classes,self.total_ds_cnt)
        # plt.tight_layout()
        # plt.gcf().subplots_adjust(bottom=0.40)
        # plt.xticks(rotation = 90)
        # plt.xlabel('Diseases')
        # plt.savefig('C:/Users/hb/Desktop/code/Influencer_learning/IL/data_preprocessing/Client{}_disease_distribution.png'.format(c_num))
        # plt.clf()

    def get_ds_cnt(self, c_num):

        raw_pos_freq = self.total_ds_cnt
        raw_neg_freq = self.total_ds_cnt.sum() - self.total_ds_cnt

        return raw_pos_freq, raw_neg_freq
            
    def compute_class_freqs(self):
        # total number of patients (rows)
        labels = self.train_val_df ## What is the shape of this???
        N = labels.shape[0]
        positive_frequencies = (labels.sum(axis = 0))/N
        negative_frequencies = 1.0 - positive_frequencies
    
        return positive_frequencies, negative_frequencies

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_train_val_df(self):

        # get the list of train_val data 
        train_val_list = self.get_train_val_list()
        print("train_va_list: ",len(train_val_list))

        train_val_df = pd.DataFrame()
        print('\nbuilding train_val_df...')
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            if filename in train_val_list:
                train_val_df = train_val_df.append(self.df.iloc[i:i+1, :])
        return train_val_df

    def __getitem__(self, index):

        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']
        row = self.new_df.iloc[index, :]
        # img = cv2.imread(row['image_links'])
        img = Image.open(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        target = torch.zeros(len(self.all_classes))
        new_target = torch.zeros(len(self.all_classes) - 1)
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1            
        if self.transform is not None:
            img = self.transform(img)

        return img, target[:14]
       
    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]

        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df
    
    def get_train_val_list(self):
        f = open("C:/Users/hamdo/Desktop/data/NIH/train_Val_list.txt", 'r')
        train_val_list = str.split(f.read(), '\n')
        return train_val_list

    def __len__(self):
        return len(self.the_chosen)
    
    def get_name(self):
        return 'NIH'

    def get_class_cnt(self):
        return 14

class NIHTestDataset(Dataset):

    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        # full dataframe including train_val and test set
        self.df = self.get_df()
        self.make_pkl_dir(config.pkl_dir_path)
        self.disease_cnt = [0]*14
        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']

        # loading the classes list
        
        with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
            self.all_classes = pickle.load(handle) 
        # get test_df
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.test_df_pkl_path)):
            self.test_df = self.get_test_df()
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'wb') as handle:
                pickle.dump(self.test_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
            print('\n{}: dumped'.format(config.test_df_pkl_path))
        else:
            # pickle load the test_df
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'rb') as handle:
                self.test_df = pickle.load(handle)

        for i in range(len(self.test_df)):
            row = self.test_df.iloc[i, :]
            labels = str.split(row['Finding Labels'], '|')
            for lab in labels:  
                lab_idx = self.all_classes.index(lab)
                if lab_idx == 14: # No Finding
                    continue
                self.disease_cnt[lab_idx] += 1

    def get_ds_cnt(self):
        return self.disease_cnt

    def __getitem__(self, index):
        row = self.test_df.iloc[index, :]
        # img = cv2.imread(row['image_links'])
        img = Image.open(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        target = torch.zeros(len(self.all_classes)) # 15
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1     
        if self.transform is not None:
            img = self.transform(img)
        return img, target[:14]

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        all_xray_df = pd.read_csv(csv_path)
        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]
        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df

    def get_test_df(self):
        # get the list of test data 
        test_list = self.get_test_list()
        test_df = pd.DataFrame()
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            if filename in test_list:
                test_df = test_df.append(self.df.iloc[i:i+1, :])
        return test_df

    def get_test_list(self):
        f = open( os.path.join('C:/Users/hamdo/Desktop/data/NIH', 'test_list.txt'), 'r')
        test_list = str.split(f.read(), '\n')
        return test_list

    def __len__(self):
        return len(self.test_df)

class NIHQualificationDataset(Dataset):

    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        # full dataframe including train_val and test set
        self.df = self.get_df()
        self.make_pkl_dir(config.pkl_dir_path)
        self.disease_cnt = [0]*14
        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']

        # loading the classes list
        with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
            self.all_classes = pickle.load(handle) 
        # get test_df
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.qualification_df_pkl_path)):
            self.test_df = self.get_test_df()
            with open(os.path.join(config.pkl_dir_path, config.qualification_df_pkl_path), 'wb') as handle:
                pickle.dump(self.test_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
            print('\n{}: dumped'.format(config.qualification_df_pkl_path))
        else:
            # pickle load the test_df
            with open(os.path.join(config.pkl_dir_path, config.qualification_df_pkl_path), 'rb') as handle:
                self.test_df = pickle.load(handle)

        for i in range(len(self.test_df)):
            row = self.test_df.iloc[i, :]
            labels = str.split(row['Finding Labels'], '|')
            for lab in labels:  
                lab_idx = self.all_classes.index(lab)
                if lab_idx == 14: # No Finding
                    continue
                self.disease_cnt[lab_idx] += 1

    def get_ds_cnt(self):
        return self.disease_cnt

    def __getitem__(self, index):
        row = self.test_df.iloc[index, :]
        # img = cv2.imread(row['image_links'])
        img = Image.open(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        target = torch.zeros(len(self.all_classes)) # 15
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1     
        if self.transform is not None:
            img = self.transform(img)
        return img, target[:14]

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        all_xray_df = pd.read_csv(csv_path)
        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]
        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df

    def get_test_df(self):
        # get the list of test data 
        test_list = self.get_test_list()
        test_df = pd.DataFrame()
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            if filename in test_list:
                test_df = test_df.append(self.df.iloc[i:i+1, :])
        return test_df

    def get_test_list(self):
        f = open( os.path.join('C:/Users/hamdo/Desktop/data/NIH', 'qualification.txt'), 'r')
        test_list = str.split(f.read(), '\n')
        return test_list

    def __len__(self):
        return len(self.test_df)

class NIHBackupDataset(Dataset):

    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        # full dataframe including train_val and test set
        self.df = self.get_df()
        self.make_pkl_dir(config.pkl_dir_path)
        self.disease_cnt = [0]*14
        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']

        # loading the classes list
        with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
            self.all_classes = pickle.load(handle) 
        # get test_df
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.backup_df_pkl_path)):
            self.test_df = self.get_test_df()
            with open(os.path.join(config.pkl_dir_path, config.backup_df_pkl_path), 'wb') as handle:
                pickle.dump(self.test_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
            print('\n{}: dumped'.format(config.backup_df_pkl_path))
        else:
            # pickle load the test_df
            with open(os.path.join(config.pkl_dir_path, config.backup_df_pkl_path), 'rb') as handle:
                self.test_df = pickle.load(handle)

        for i in range(len(self.test_df)):
            row = self.test_df.iloc[i, :]
            labels = str.split(row['Finding Labels'], '|')
            for lab in labels:  
                lab_idx = self.all_classes.index(lab)
                if lab_idx == 14: # No Finding
                    continue
                self.disease_cnt[lab_idx] += 1

    def get_ds_cnt(self):
        return self.disease_cnt

    def __getitem__(self, index):
        row = self.test_df.iloc[index, :]
        # img = cv2.imread(row['image_links'])
        img = Image.open(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        target = torch.zeros(len(self.all_classes)) # 15
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1     
        if self.transform is not None:
            img = self.transform(img)
        return img, target[:14]

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        all_xray_df = pd.read_csv(csv_path)
        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]
        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df

    def get_test_df(self):
        # get the list of test data 
        test_list = self.get_test_list()
        test_df = pd.DataFrame()
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            if filename in test_list:
                test_df = test_df.append(self.df.iloc[i:i+1, :])
        return test_df

    def get_test_list(self):
        f = open( os.path.join('C:/Users/hamdo/Desktop/data/NIH', 'backup_list.txt'), 'r')
        test_list = str.split(f.read(), '\n')
        return test_list

    def __len__(self):
        return len(self.test_df)

class CelebADataset(Dataset):

    def __init__(self, transform = None, indices = None):
        
        csv_path = "C:/Users/hamdo/Desktop/data/celebA/list_attr_celeba.csv"
        self.dir = "C:/Users/hamdo/Desktop/data/celebA/img_align_celeba/img_align_celeba/"
        self.transform = transform[0]

        self.all_data = pd.read_csv(csv_path)
        self.all_data = self.all_data.replace(to_replace=-1,value=0)
        self.selecte_data = self.all_data.iloc[indices, :]
        self.class_num = 40

    def __getitem__(self, index):

        row = self.selecte_data.iloc[index, :]
        # img = cv2.imread(self.dir + row['Path'])
        img = pilimg.open(self.dir + row['image_id'])
        label = torch.FloatTensor(row[1:])
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.selecte_data)

class ChexpertTrainDataset(Dataset):

    def __init__(self,c_num, transform = None, indices = None):
        
        csv_path = "C:/Users/hb/Desktop/data/CheXpert-v1.0-small/selected_train.csv"
        self.dir = "C:/Users/hb/Desktop/data/"
        self.transform = transform

        self.all_data = pd.read_csv(csv_path)
        self.selecte_data = self.all_data.iloc[indices, :]
        self.class_num = 10
        self.all_classes = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Fracture']
        
        self.total_ds_cnt = self.get_total_cnt()
        self.total_ds_cnt = np.array(self.total_ds_cnt)
        # Normalize the imbalance
        self.imbalance = 0
        difference_cnt = self.total_ds_cnt - self.total_ds_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] * difference_cnt[i]        
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] / difference_cnt.sum()
        # Calculate the level of imbalnce
        difference_cnt -= difference_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = (difference_cnt[i] * difference_cnt[i])
    
        self.imbalance = 1 / difference_cnt.sum()

        # Plot the disease distribution
        plt.figure(figsize=(8,4))
        plt.title('Client{} Disease Distribution'.format(c_num), fontsize=20)
        plt.bar(self.all_classes,self.total_ds_cnt)
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.40)
        plt.xticks(rotation = 90)
        plt.xlabel('Diseases')
        plt.savefig('C:/Users/hb/Desktop/code/3.FedBalance_mp/data/ChexPert/Client{}_disease_distribution.png'.format(c_num))
        plt.clf()

    def __getitem__(self, index):

        row = self.selecte_data.iloc[index, :]
        # img = cv2.imread(self.dir + row['Path'])
        img = pilimg.open(self.dir + row['Path'])
        label = torch.FloatTensor(row[2:])
        gray_img = self.transform(img)
        return torch.cat([gray_img,gray_img,gray_img], dim = 0), label

    def __len__(self):
        return len(self.selecte_data)

    def get_total_cnt(self):
        total_ds_cnt = [0] * self.class_num
        for i in range(len(self.selecte_data)):
            row = self.selecte_data.iloc[i, 2:]
            for j in range(len(row)):
                total_ds_cnt[j] += int(row[j])
        return total_ds_cnt

    def get_ds_cnt(self):

        raw_pos_freq = self.total_ds_cnt
        raw_neg_freq = self.total_ds_cnt.sum() - self.total_ds_cnt

        return raw_pos_freq, raw_neg_freq

    def get_name(self):
        return 'CheXpert'

    def get_class_cnt(self):
        return 10

class ChexpertQualificationDataset(Dataset):

    def __init__(self,transform = None, indices = None):
        
        csv_path = "C:/Users/hb/Desktop/data/CheXpert-v1.0-small/selected_train.csv"
        self.dir = "C:/Users/hb/Desktop/data/"
        self.transform = transform

        self.all_data = pd.read_csv(csv_path)
        self.selecte_data = self.all_data.iloc[indices, :]
        self.class_num = 10
        self.all_classes = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Fracture']
        
        self.total_ds_cnt = self.get_total_cnt()
        self.total_ds_cnt = np.array(self.total_ds_cnt)
        # Normalize the imbalance
        self.imbalance = 0
        difference_cnt = self.total_ds_cnt - self.total_ds_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] * difference_cnt[i]        
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] / difference_cnt.sum()
        # Calculate the level of imbalnce
        difference_cnt -= difference_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = (difference_cnt[i] * difference_cnt[i])
    
        self.imbalance = 1 / difference_cnt.sum()

    def __getitem__(self, index):

        row = self.selecte_data.iloc[index, :]
        # img = cv2.imread(self.dir + row['Path'])
        img = pilimg.open(self.dir + row['Path'])
        label = torch.FloatTensor(row[2:])
        gray_img = self.transform(img)
        return torch.cat([gray_img,gray_img,gray_img], dim = 0), label

    def __len__(self):
        return len(self.selecte_data)

    def get_total_cnt(self):
        total_ds_cnt = [0] * self.class_num
        for i in range(len(self.selecte_data)):
            row = self.selecte_data.iloc[i, 2:]
            for j in range(len(row)):
                total_ds_cnt[j] += int(row[j])
        return total_ds_cnt

    def get_ds_cnt(self):

        raw_pos_freq = self.total_ds_cnt
        raw_neg_freq = self.total_ds_cnt.sum() - self.total_ds_cnt

        return raw_pos_freq, raw_neg_freq

    def get_name(self):
        return 'CheXpert'

    def get_class_cnt(self):
        return 10

class ChexpertTestDataset(Dataset):

    def __init__(self, transform = None):
        
        csv_path = "C:/Users/hb/Desktop/data/CheXpert-v1.0-small/selected_test.csv"
        self.dir = "C:/Users/hb/Desktop/data/"
        self.transform = transform

        self.all_data = pd.read_csv(csv_path)
        self.selecte_data = self.all_data.iloc[:, :]
        # self.selecte_data.to_csv("C:/Users/hb/Desktop/data/CheXpert-v1.0-small/selected_data.csv")
        self.class_num = 10

    def __getitem__(self, index):

        row = self.selecte_data.iloc[index, :]
        img = pilimg.open(self.dir + row['Path'])
        label = torch.FloatTensor(row[2:])
        gray_img = self.transform(img)

        return torch.cat([gray_img,gray_img,gray_img], dim = 0), label

    def get_ds_cnt(self):
        total_ds_cnt = [0] * self.class_num
        for i in range(len(self.selecte_data)):
            row = self.selecte_data.iloc[i, 2:]
            for j in range(len(row)):
                total_ds_cnt[j] += int(row[j])
        return total_ds_cnt

    def __len__(self):
        return len(self.selecte_data)

class BraTS2021TrainLoader(Dataset): # custom dataset

    def BraTS2021loader(self, index): # index = number
        
        brain = np.load(self.dir + "/imgs/" + str(index) + ".npy")
        mask = np.load(self.dir + "/labels/" + str(index) + ".npy")                  

        mean = np.mean(brain)
        std = np.std(brain)

        if mean == 0 or std == 0 :
            transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            ])
        else :
            transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=False),
            ]) 
        

        transform2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        ])

        kidney = transform1(brain)
        mask = transform2(mask)

        # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
        # ax1.imshow(kidney[0])
        # ax1.set_title('input')
        # ax2.imshow(mask[0])
        # ax2.set_title('mask')
        # plt.show()

        return kidney, mask

    def __init__(self, dir, participant_num, indices):
        
        self.dir = dir + '/Training/participant' + str(participant_num)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index): 
        images, masks= self.BraTS2021loader(self.indices[index])

        return images, masks
    
class BraTS2021QualificationLoader(Dataset): # custom dataset

    def BraTS2021loader(self, index): # index = number
        
        brain = np.load(self.dir + "/imgs/" + str(index) + ".npy")
        mask = np.load(self.dir + "/labels/" + str(index) + ".npy")                  

        mean = np.mean(brain)
        std = np.std(brain)

        if mean == 0 or std == 0 :
            transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            ])
        else :
            transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=False),
            ]) 
        

        transform2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        ])

        kidney = transform1(brain)
        mask = transform2(mask)

        # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
        # ax1.imshow(kidney[0])
        # ax1.set_title('input')
        # ax2.imshow(mask[0])
        # ax2.set_title('mask')
        # plt.show()

        return kidney, mask

    def __init__(self, dir):

        indices = list(range(77500))
        random.shuffle(indices)
        self.indices = indices[int(0.4 * len(indices)):int(0.7 * len(indices))]
        self.dir = dir + "/Qualification"
        print("Qualificiation dataset : ",len(self.indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index): 
        images, masks= self.BraTS2021loader(self.indices[index])

        return images, masks
    
class BraTS2021TestLoader(Dataset): # custom dataset

    def BraTS2021loader(self, index): # index = number
        
        brain = np.load(self.dir + "/imgs/" + str(index) + ".npy")
        mask = np.load(self.dir + "/labels/" + str(index) + ".npy")                  

        mean = np.mean(brain)
        std = np.std(brain)

        if mean == 0 or std == 0 :
            transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            ])
        else :
            transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=False),
            ]) 
        

        transform2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        ])

        kidney = transform1(brain)
        mask = transform2(mask)

        # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
        # ax1.imshow(kidney[0])
        # ax1.set_title('input')
        # ax2.imshow(mask[0])
        # ax2.set_title('mask')
        # plt.show()

        return kidney, mask

    def __init__(self, dir):
        
        self.indices = list(range(77500))
        random.shuffle(self.indices)
        self.indices = self.indices[:int(0.1 * len(self.indices))]
        self.dir = dir + "/Test"
        print("Test dataset : ",len(self.indices))

    def __len__(self):
        # len(os.listdir(self.dir + '/imgs'))
        return len(self.indices)

    def __getitem__(self, index): 
        images, masks= self.BraTS2021loader(self.indices[index])

        return images, masks