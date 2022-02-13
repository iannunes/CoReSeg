from os import listdir
from os.path import isfile, join
import os
from skimage import io
from skimage import util
from skimage import transform
from sklearn import metrics
from torch.utils import data
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from argparse import ArgumentParser
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from joblib import Parallel, delayed

def load_my_state_dict(model1, model2):
    own_state = model1.state_dict()
    counter=0
    total_counter = 0
    for name, param in model2.items():
        total_counter+=1
        if name not in own_state:
             continue
        if isinstance(param, nn.Parameter):
            param = param.data
        counter+=1
        own_state[name].copy_(param)
    print(counter,' of ',total_counter, " - parameters loaded") 
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def get_loaders(loader_args, train_overlap=True, val_overlap=True, valtrain_overlap=False, test_overlap=True, val_threshold_overlap=True):
    batch_size = 1
    train_loader=[]
    valtrain_loader=[]
    test_loader=[]
    val_threshold_loader=[]
    
    if loader_args.train_model:
        train_set = ListDatasetC2AE(loader_args.dataset, 'Train', (loader_args.h_size, loader_args.w_size), 'statistical', loader_args.hidden, overlap=train_overlap, use_dsm=True, dataset_path=loader_args.datasets_path+'/', select_non_match=loader_args.select_non_match, ignore_others_non_match=loader_args.ignore_others_non_match, inmemory=True, local_batch_size=loader_args.local_batch_size, backgroundclasses=loader_args.backgroundclasses)
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=loader_args.workers, shuffle=True)

        #ValidateTrain
        valtrain_set = ListDatasetC2AE(loader_args.dataset, 'Thresholds', (loader_args.h_size, loader_args.w_size), 'statistical', loader_args.hidden, overlap=valtrain_overlap, use_dsm=True, dataset_path=loader_args.datasets_path+'/', select_non_match=loader_args.select_non_match, ignore_others_non_match=loader_args.ignore_others_non_match, inmemory=True, local_batch_size=loader_args.local_batch_size, backgroundclasses=loader_args.backgroundclasses)
        valtrain_loader = DataLoader(valtrain_set, batch_size=batch_size, num_workers=loader_args.workers, shuffle=False)

    if loader_args.prep_thresholds:
        val_threshold_set = ListDatasetC2AE(loader_args.dataset, 'Thresholds', (loader_args.h_size, loader_args.w_size), 'statistical', loader_args.hidden, overlap=val_threshold_overlap, use_dsm=True, dataset_path=loader_args.datasets_path+'/', select_non_match=loader_args.select_non_match, ignore_others_non_match=loader_args.ignore_others_non_match, inmemory=True, local_batch_size=loader_args.local_batch_size, backgroundclasses=loader_args.backgroundclasses)
        val_threshold_loader = DataLoader(val_threshold_set, batch_size=batch_size, num_workers=loader_args.workers, shuffle=False)
    
    if loader_args.prep_eval or loader_args.eval_model==True:
        test_set = ListDatasetC2AE(loader_args.dataset, 'Test', (loader_args.h_size, loader_args.w_size), 'statistical', loader_args.hidden, overlap=test_overlap, use_dsm=True, dataset_path=loader_args.datasets_path+'/', select_non_match=loader_args.select_non_match, ignore_others_non_match=loader_args.ignore_others_non_match, inmemory=True, local_batch_size=loader_args.local_batch_size, backgroundclasses=loader_args.backgroundclasses)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=loader_args.workers, shuffle=False)
    if loader_args.prep_eval==False and loader_args.eval_model==True:
        test_loader=len(test_loader)
    
    return train_loader,valtrain_loader,test_loader,val_threshold_loader

class ListDatasetC2AE(data.Dataset):
    
    def __init__(self, dataset, mode, crop_size=(224,224), normalization='minmax', hidden_classes=None, overlap=False, use_dsm=False, dataset_path='../', select_non_match='none', ignore_others_non_match=False, verbose=False, inmemory=False, local_batch_size=2, backgroundclasses=[]): #none or random
        
        # Initializing variables.
        self.dataset_path = dataset_path
        self.root =  dataset_path + dataset + '/'
        self.dataset = dataset
        self.mode = mode.strip(' ')
        self.crop_size = crop_size
        self.normalization = normalization
        self.hidden_classes = hidden_classes
        self.overlap = overlap
        self.use_dsm = use_dsm
        self.select_non_match = select_non_match
        self.ignore_others_non_match = ignore_others_non_match
        self.verbose=verbose
        self.inmemory=inmemory
        self.local_batch_size = local_batch_size
        self.backgroundclasses = backgroundclasses
        
        if self.dataset == 'GRSS':
            self.num_classes = 21
        else:
            self.num_classes = 5
        self.dataset_size_factor = 1
            
        if self.hidden_classes is not None:
            self.n_classes = self.num_classes - len(hidden_classes)
        else:
            self.n_classes = self.num_classes
            
        #print('self.n_classes', self.n_classes)
        #print('self.hidden_classes', self.hidden_classes)
        
        # Creating list of paths.
        self.imgs = self.make_dataset()

        # Check for consistency in list.
        
        if self.dataset == 'GRSS':
            if len(self.img_single) == 0:
                raise (RuntimeError('Found 0 images, please check the data set'))
        else:
            if len(self.imgs) == 0:
                raise (RuntimeError('Found 0 images, please check the data set'))
                
    def make_dataset(self):
        
        # Making sure the mode is correct.
        assert self.mode in ['Train', 'Test', 'Validate', 'ValidateTrain', 'Thresholds','Thresholds_Train']
        
        # Setting string for the mode.
        
        img_folder = self.mode
        if self.mode=='Thresholds':
            img_folder = 'ValidateTrain'
        if self.mode=='Thresholds_Train':
            img_folder = 'Train'
        
        img_dir = os.path.join(self.root, img_folder, 'JPEGImages')
        msk_dir = os.path.join(self.root, img_folder, 'Masks')
        
        if self.use_dsm:
            dsm_dir = os.path.join(self.root, img_folder, 'NDSM')
            
        if self.dataset == 'GRSS':
            # Presetting ratios across GRSS channels and labels.
            self.rgb_hsi_ratio = 20
            self.dsm_hsi_ratio = 2
            self.msk_hsi_ratio = 2
            
            self.rgb_msk_ratio = 10
            
            self.hsi_patch_size = 500
            self.rgb_patch_size = self.rgb_hsi_ratio * self.hsi_patch_size
            self.dsm_patch_size = self.dsm_hsi_ratio * self.hsi_patch_size
            self.msk_patch_size = self.msk_hsi_ratio * self.hsi_patch_size
            
            if 'Thresholds' == self.mode:
                self.mode = 'Validate'
            
            if self.mode == 'Train' or self.mode == 'Validate':
                # Reading images.
                self.img_single = io.imread(os.path.join(self.root, 'Train', 'Images', 'rgb_clipped.tif')).astype(np.uint8)
                self.msk_single = io.imread(os.path.join(self.root, 'Train', 'Masks', '2018_IEEE_GRSS_DFC_GT_TR.tif')).astype(np.int64)
                #print('self.msk_single - ',np.unique(self.msk_single, return_counts=True))
                if self.use_dsm:
                    self.dsm_single = io.imread(os.path.join(self.root, 'Train', 'DSM', 'dsm_clipped.tif'))
                    self.q0001 = -21.208378 # q0001 precomputed from training set.
                    self.q9999 = 41.01488   # q9999 precomputed from training set.
                    self.dsm_single = np.clip(self.dsm_single, self.q0001, self.q9999)
                    self.dsm_single = (self.dsm_single - self.dsm_single.min()) / (self.dsm_single.max() - self.dsm_single.min())
                    self.dsm_single *= 255.0
                    
                # train or validate mask (1 = train, 2 = validate
                train_validate = np.zeros(self.msk_single.shape,dtype=np.int8)
                train_validate[train_validate==0] = 1
                train_validate[50:600,:] = 2
                train_validate[50:600,2700:4000] = 1
                train_validate[50:600,1500:2300] = 1
                train_validate[50:600,0:500] = 1
                
                if self.mode == 'Train':
                    train_validate[train_validate==2] = 0
                else:
                    train_validate = train_validate-1
                
                #print(np.unique(self.msk_single,return_counts=True))
                
                self.msk_single = self.msk_single*train_validate
                
                #print(np.unique(self.msk_single,return_counts=True))
                
                #if self.mode == 'Validate':
                #    y0, y1, x0, x1 = trim_coords(train_validate)
                #    self.img_single = self.img_single[y0:y1, x0:x1]
                #    self.msk_single = self.msk_single[y0:y1, x0:x1]
                #    if self.use_dsm:
                #        self.dsm_single = self.dsm_single[y0:y1, x0:x1]
                        
            elif self.mode == 'Test':
                
                # Reading images.
                self.img_single = io.imread(os.path.join(self.root, 'Test', 'Images', 'rgb_merged.tif')).astype(np.uint8)
                self.msk_single = io.imread(os.path.join(self.root, 'Test', 'Masks', 'Test_Labels_osr.tif'))[:,:,0].astype(np.int64)
                self.msk_single[self.msk_single == 100] = 0
                if self.use_dsm:
                    self.dsm_single = io.imread(os.path.join(self.root, 'Test', 'DSM', 'UH17c_GEF051.tif'))
                    self.q0001 = -21.208378 # q0001 precomputed from training set.
                    self.q9999 = 41.01488   # q9999 precomputed from training set.
                    self.dsm_single = np.clip(self.dsm_single, self.q0001, self.q9999)
                    self.dsm_single = (self.dsm_single - self.dsm_single.min()) / (self.dsm_single.max() - self.dsm_single.min())
                    self.dsm_single *= 255.0
            
            unique, counts = np.unique(self.msk_single, return_counts=True)
            #print(unique)
            #print(counts)
            
            self.msk_single, self.msk_true_single = self.shift_labels(self.msk_single)
            
            unique, counts = np.unique(self.msk_single, return_counts=True)
            #print(unique)
            #print(counts)
            valid_counts = counts[:-1] if self.mode == 'Train' else counts[:-2] # Removing UUC.
            self.weights = (valid_counts.max() / valid_counts).tolist()
            #print('weights', self.weights)
            #print('len weights', len(self.weights))
            
            #print('img_single', self.img_single.shape)
            #print('dsm_single', self.dsm_single.shape)
            #print('msk_single', self.msk_single.shape)
            #print('msk_true_single', self.msk_true_single.shape)            
            return
        
        else:
            
            # Vaihingen and Potsdam.
            if self.mode == 'Validate':
                img_dir = os.path.join(self.root, 'Train', 'JPEGImages')
                msk_dir = os.path.join(self.root, 'Train', 'Masks')
                if self.use_dsm:
                    dsm_dir = os.path.join(self.root, 'Train', 'NDSM')

            data_list = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])

            # Creating list containing image and ground truth paths.
            self.data = []
            items = []
            if self.dataset == 'Vaihingen':
                for it in data_list:
                    item = (
                        os.path.join(img_dir, it),
                        os.path.join(msk_dir, it),
                        os.path.join(dsm_dir, it.replace('top_mosaic_09cm_area', 'dsm_09cm_matching_area').replace('.tif', '_normalized.jpg'))
                    )
                    items.append(item)
            elif self.dataset == 'Potsdam':
                for it in data_list:
                    item = (
                        os.path.join(img_dir, it),
                        os.path.join(msk_dir, it.replace('_IRRG.tif', '_label_noBoundary.tif')),
                        os.path.join(dsm_dir, it.replace('top_potsdam_', 'dsm_potsdam_').replace('_IRRG.tif', '_normalized_lastools.jpg'))
                    )
                    items.append(item)
            print('loader in memory: ',self.inmemory)
            if self.inmemory:    
                for i in range(0,len(items)):
                    img_path, msk_path, dsm_path = items[i]

                    #print(img_path)
                    #print(msk_path)
                    #print(dsm_path)
                    aa = io.imread(img_path)
                    bb = io.imread(msk_path)
                    #print(np.unique(bb,return_counts=True))
                    cc = io.imread(dsm_path)
                    item = (aa,bb,cc)

                    self.data.append(item)

            # Returning list.
            return items
    
    def random_crops(self, img, msk, msk_true, n_crops):
        
        img_crop_list = []
        msk_crop_list = []
        msk_true_crop_list = []
        
        rand_fliplr = np.random.random() > 0.50
        rand_flipud = np.random.random() > 0.50
        rand_rotate = np.random.random()
        #print('r1 - ', datetime.now())
        i=0
        background = self.num_classes - len(self.hidden_classes) - len(self.backgroundclasses)
        #print('background:',background)
        while i < n_crops:
           
            rand_y = np.random.randint(msk.shape[0] - self.crop_size[0])
            rand_x = np.random.randint(msk.shape[1] - self.crop_size[1])

            img_patch = img[rand_y:(rand_y + self.crop_size[0]),
                            rand_x:(rand_x + self.crop_size[1])]
            msk_patch = msk[rand_y:(rand_y + self.crop_size[0]),
                            rand_x:(rand_x + self.crop_size[1])]
            msk_true_patch = msk_true[rand_y:(rand_y + self.crop_size[0]),
                                      rand_x:(rand_x + self.crop_size[1])]
            
            masked_pixels = len(msk_patch[msk_patch!=background])/(msk_patch.shape[0]*msk_patch.shape[1])
            if masked_pixels < 0.001:
                continue
                
            i+=1
            
            if rand_fliplr:
                img_patch = np.fliplr(img_patch)
                msk_patch = np.fliplr(msk_patch)
                msk_true_patch = np.fliplr(msk_true_patch)
            if rand_flipud:
                img_patch = np.flipud(img_patch)
                msk_patch = np.flipud(msk_patch)
                msk_true_patch = np.flipud(msk_true_patch)
            
            if rand_rotate < 0.25:
                img_patch = transform.rotate(img_patch, 270, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 270, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 270, order=0, preserve_range=True)
            elif rand_rotate < 0.50:
                img_patch = transform.rotate(img_patch, 180, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 180, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 180, order=0, preserve_range=True)
            elif rand_rotate < 0.75:
                img_patch = transform.rotate(img_patch, 90, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 90, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 90, order=0, preserve_range=True)
                
            img_patch = img_patch.astype(np.float32)
            msk_patch = msk_patch.astype(np.int64)
            msk_true_patch = msk_true_patch.astype(np.int64)
            
            img_crop_list.append(img_patch)
            msk_crop_list.append(msk_patch)
            msk_true_crop_list.append(msk_true_patch)
        
        img = np.asarray(img_crop_list)
        msk = np.asarray(msk_crop_list)
        msk_true = np.asarray(msk_true_crop_list)
        #print('r2 - ', datetime.now())
        return img, msk, msk_true
        
        
    def test_crops(self, img, msk, msk_true):
        
        n_channels = 3
        if self.use_dsm:
            n_channels = 4
        if self.overlap:
            w_img = util.view_as_windows(img,
                                         (self.crop_size[0], self.crop_size[1], n_channels),
                                         (self.crop_size[0] // 2, self.crop_size[1] // 2, n_channels)).squeeze()
            w_msk = util.view_as_windows(msk,
                                         (self.crop_size[0], self.crop_size[1]),
                                         (self.crop_size[0] // 2, self.crop_size[1] // 2))
            w_msk_true = util.view_as_windows(msk_true,
                                              (self.crop_size[0], self.crop_size[1]),
                                              (self.crop_size[0] // 2, self.crop_size[1] // 2))
        else:
            w_img = util.view_as_blocks(img, (self.crop_size[0], self.crop_size[1], n_channels)).squeeze()
            w_msk = util.view_as_blocks(msk, (self.crop_size[0], self.crop_size[1]))
            w_msk_true = util.view_as_blocks(msk_true, (self.crop_size[0], self.crop_size[1]))
        
        return w_img, w_msk, w_msk_true
        
    def shift_labels(self, msk):
        
        msk_true = np.copy(msk)
        #print('shift_labels - 1')
        #print(np.unique(msk_true, return_counts=True))
            
        if self.dataset == 'Vaihingen' or self.dataset == 'Potsdam':
            
            # Shifting clutter/background to unknown on labels.
            msk[msk == 5] = 2000
            msk[msk == 6] = 2000
        elif self.dataset == 'GRSS':
            
            self.backgroundclasses.sort(reverse=True)            
            for k in self.backgroundclasses:
                msk[msk == k] = 2000
                pos = (msk > k) & (msk < 2000)
                msk[pos] -= 1

            new_hidden = []
            for h_c in sorted(self.hidden_classes):
                if h_c > 0 and h_c < 7:
                    new_hidden.append(h_c - 1)
                elif h_c > 7:
                    new_hidden.append(h_c - 2)
            assert len(self.hidden_classes) == len(new_hidden)
            self.hidden_classes = new_hidden
        
        #print('shift_labels - 2')
        #print(np.unique(msk_true, return_counts=True))
        
        cont = 0
        for h_c in sorted(self.hidden_classes):
            
            #print('Hidden %d' % (h_c))
            msk[msk == h_c - cont] = 1000
            for c in range(h_c - cont + 1, self.num_classes):
                #print('    Class %d -> %d' % (c, c - 1))
                msk[msk == c] = c - 1
                msk_true[msk_true == c] = c - 1
            cont = cont + 1
        
        #print('shift_labels - 3')
        #print(np.unique(msk_true, return_counts=True))
        
        if self.dataset == 'GRSS':
            msk_true = np.copy(msk)
            msk_true[msk == 1000] = self.num_classes - len(self.hidden_classes) - len(self.backgroundclasses)
            msk_true[msk == 2000] = self.num_classes - 1
            msk[msk >= 1000] = self.num_classes - len(self.hidden_classes) - len(self.backgroundclasses)
        else:
            msk_true[msk == 1000] = self.num_classes - len(self.hidden_classes)
            msk_true[msk == 2000] = self.num_classes
            msk[msk >= 1000] = self.num_classes - len(self.hidden_classes)
        
        #print('shift_labels - msk after', np.unique(msk, return_counts=True))
        #print('shift_labels - msk_true after', np.unique(msk_true, return_counts=True))
        #sys.exit()
        return msk, msk_true
    
    def mask_to_class(self, msk):
        
        msk = msk.astype(np.int64)
        new = np.zeros((msk.shape[0], msk.shape[1]), dtype=np.int64)
        
        msk = msk // 255
        msk = msk * (1, 7, 49)
        msk = msk.sum(axis=2)
        
        new[msk == 1 + 7 + 49] = 0 # Street.
        new[msk ==         49] = 1 # Building.
        new[msk ==     7 + 49] = 2 # Grass.
        new[msk ==     7     ] = 3 # Tree.
        new[msk == 1 + 7     ] = 4 # Car.
        new[msk == 1         ] = 5 # Surfaces.
        new[msk == 0         ] = 6 # Boundaries.
        
        return new        
     
    def __getitem__(self, index):
        x = datetime.now()
        s = int(x.strftime('%d%Y%m%w%M%S%f%H'))%(2**31)
        s = s // (index+1)
        np.random.seed(s)
        
        tic = datetime.now()
        
        img_raw = None
        msk_raw = None
        dsm_raw = None
        
        if self.dataset == 'GRSS':
            if self.mode == 'Train':

                offset_rgb = np.random.randint(self.rgb_msk_ratio, size=2) if self.mode == 'Train' else (self.rgb_msk_ratio // 2, self.rgb_msk_ratio // 2)
                
                img_raw = self.img_single[offset_rgb[0]::self.rgb_msk_ratio,
                                          offset_rgb[1]::self.rgb_msk_ratio]
                msk_raw = self.msk_single
                msk_true_raw = self.msk_true_single
                if self.use_dsm:
                    dsm_raw = self.dsm_single
                
                assert img_raw.shape[0] == dsm_raw.shape[0] and\
                       img_raw.shape[0] == msk_raw.shape[0] and\
                       img_raw.shape[0] == msk_true_raw.shape[0] and\
                       img_raw.shape[1] == dsm_raw.shape[1] and\
                       img_raw.shape[1] == msk_raw.shape[1] and\
                       img_raw.shape[1] == msk_true_raw.shape[1], 'Shape Inconsistency: rgb = ' + str(img_raw.shape) + ', dsm = ' + str(dsm_raw.shape) + ', msk = ' + str(msk_raw.shape) + ', msk_true = ' + str(msk_true_raw.shape)
                
            else:
                
                img_raw = self.img_single[self.rgb_msk_ratio // 2::self.rgb_msk_ratio,
                                          self.rgb_msk_ratio // 2::self.rgb_msk_ratio]
                msk_raw = self.msk_single
                msk_true_raw = self.msk_true_single
                if self.use_dsm:
                    dsm_raw = self.dsm_single
                    
                assert img_raw.shape[0] == dsm_raw.shape[0] and\
                       img_raw.shape[0] == msk_raw.shape[0] and\
                       img_raw.shape[0] == msk_true_raw.shape[0] and\
                       img_raw.shape[1] == dsm_raw.shape[1] and\
                       img_raw.shape[1] == msk_raw.shape[1] and\
                       img_raw.shape[1] == msk_true_raw.shape[1], 'Shape Inconsistency: rgb = ' + str(img_raw.shape) + ', dsm = ' + str(dsm_raw.shape) + ', msk = ' + str(msk_raw.shape) + ', msk_true = ' + str(msk_true_raw.shape)
                
             
            if len(img_raw.shape) == 2:
                img_raw = color.gray2rgb(img_raw)
        
            if self.use_dsm:
                img = np.full((img_raw.shape[0] + self.crop_size[0] - (img_raw.shape[0] % self.crop_size[0]),
                               img_raw.shape[1] + self.crop_size[1] - (img_raw.shape[1] % self.crop_size[1]),
                               img_raw.shape[2] + 1),
                              fill_value=0.0,
                              dtype=np.float32)
            else:
                img = np.full((img_raw.shape[0] + self.crop_size[0] - (img_raw.shape[0] % self.crop_size[0]),
                               img_raw.shape[1] + self.crop_size[1] - (img_raw.shape[1] % self.crop_size[1]),
                               img_raw.shape[2]),
                              fill_value=0.0,
                              dtype=np.float32)

            img[:img_raw.shape[0], :img_raw.shape[1], :img_raw.shape[2]] = img_raw
            if self.use_dsm:
                img[:dsm_raw.shape[0], :dsm_raw.shape[1], -1] = dsm_raw
                
            
            msk = np.full((msk_raw.shape[0] + self.crop_size[0] - (msk_raw.shape[0] % self.crop_size[0]),
                           msk_raw.shape[1] + self.crop_size[1] - (msk_raw.shape[1] % self.crop_size[1])),
                          fill_value=(self.num_classes - len(self.hidden_classes) - len(self.backgroundclasses)),
                          dtype=np.int64)
            msk[:msk_raw.shape[0], :msk_raw.shape[1]] = msk_raw
            
            msk_true = np.full((msk_true_raw.shape[0] + self.crop_size[0] - (msk_true_raw.shape[0] % self.crop_size[0]),
                                msk_true_raw.shape[1] + self.crop_size[1] - (msk_true_raw.shape[1] % self.crop_size[1])),
                               fill_value=(self.num_classes - 1),
                               dtype=np.int64)
            msk_true[:msk_true_raw.shape[0], :msk_true_raw.shape[1]] = msk_true_raw
            
            img = (img / 255) - 0.5
            
            if self.mode == 'Train':
            
                #img, msk, msk_true = self.random_crops(img, msk, msk_true, 4)
                #img = np.transpose(img, (0, 3, 1, 2))               
                
                if self.select_non_match=='random':
                    local_random_crops = 50
                    if self.local_batch_size*40>50:
                        local_random_crops = self.local_batch_size*40

                    img, msk, msk_true = self.random_crops(img, msk, msk_true, local_random_crops)
                    toc = datetime.now()
                    if self.verbose:
                        print('getitem random1: %.2f',(toc-tic))
                    tic = datetime.now()
                    search_imgs = img[self.local_batch_size:,:,:,:]
                    search_msks = msk[self.local_batch_size:,:,:]
                    search_msks_true = msk_true[self.local_batch_size:,:,:]

                    img_m = np.copy(img[:self.local_batch_size,:,:,:])
                    msk_m = np.copy(msk[:self.local_batch_size,:,:])
                    msk_true_m =  np.copy(msk_true[:self.local_batch_size,:,:])

                    img_nm = np.copy(search_imgs[:self.local_batch_size,:,:,:])
                    msk_nm = np.copy(search_msks[:self.local_batch_size,:,:])
                    msk_true_nm =  np.copy(search_msks_true[:self.local_batch_size,:,:])

                    for ii in range(0, self.local_batch_size):
                        match = np.sum(msk_m[ii] == msk_m[ii])
                        for jj in range(0, search_imgs.shape[0]):
                            match_temp = np.sum(msk_m[ii] == search_msks[jj])
                            if match_temp<match:
                                img_nm[ii] = search_imgs[jj]
                                msk_nm[ii] = search_msks[jj]
                                msk_true_nm[ii] = search_msks_true[jj]
                                match = match_temp
                    toc = datetime.now()
                    if self.verbose:
                        print('getitem random2: %.2f',(toc-tic))
                    tic = datetime.now()
                else:
                    img, msk, msk_true = self.random_crops(img, msk, msk_true, self.local_batch_size) 
                    img_m = img
                    msk_m = msk
                    msk_true_m = msk_true

                    img_nm = img
                    msk_nm = np.copy(msk_m)
                    msk_true_nm = np.copy(msk_true_m)
                    max_class = self.num_classes - 1

                    if self.ignore_others_non_match:
                        max_class = self.num_classes - 2   

                    for b in range(0,msk_nm.shape[0]):
                        new_classes = np.random.permutation(max_class+1)

                        next_permutation = True
                        counter_p=0
                        while next_permutation:
                            next_permutation = False
                            counter_p+=1
                            new_classes = np.random.permutation(max_class+1)
                            for i in range(0,max_class + 1):
                                if i==new_classes[i]:
                                    next_permutation=True
                                    break

                        for i in range(0,max_class + 1):
                            msk_nm[b][msk_m[b]==i] = new_classes[i]
                        for i in range(0,max_class + 1):
                            msk_true_nm[b][msk_true_m[b]==i] = new_classes[i]
                
                 #print("match % - ",str(100*match/(msk_nm.shape[1]*msk_nm.shape[2]*msk_nm.shape[0])))
                img_m = np.transpose(img_m, (0, 3, 1, 2))
                img_nm = np.transpose(img_nm, (0, 3, 1, 2))
                img_m = torch.from_numpy(img_m)
                msk_m = torch.from_numpy(msk_m)
                msk_true_m = torch.from_numpy(msk_true_m)
                img_nm = torch.from_numpy(img_nm)
                msk_nm = torch.from_numpy(msk_nm)
                msk_true_nm = torch.from_numpy(msk_true_nm)

                toc = datetime.now()
                if self.verbose:
                    print('getitem ending: %.2f',(toc-tic))
                tic = datetime.now()
                sys.stdout.flush()

                return img_m, msk_m, msk_true_m, img_nm, msk_nm, msk_true_nm

            elif self.mode == 'Validate' or self.mode == 'Test':

                #print('msk - ',np.unique(self.msk_single,return_counts=True))
                #print('gts - ',np.unique(self.msk_true_single,return_counts=True))
                
                img, msk, msk_true = self.test_crops(img, msk, msk_true)

                img = np.transpose(img, (0, 1, 4, 2, 3))
                msk = np.transpose(msk, (0, 1, 2, 3))
                msk_true = np.transpose(msk_true, (0, 1, 2, 3))
            
            #print('msk - ',np.unique(msk,return_counts=True))
            #print('gts - ',np.unique(msk_true,return_counts=True))
            img = torch.from_numpy(img)
            msk = torch.from_numpy(msk)
            msk_true = torch.from_numpy(msk_true)
            # Returning to iterator.
            return img, msk, msk_true, 'img.tif'
            
        else:
            ii=index
            if self.mode == 'Train':
                ii=index%(len(self.imgs))        

            # Reading items from list.
            if self.use_dsm:
                img_path, msk_path, dsm_path = self.imgs[ii]
                if self.inmemory:
                    img_raw, msk_raw, dsm_raw = self.data[ii]
                    #print(np.unique(msk_raw,return_counts=True))
                else:
                    img_raw = io.imread(img_path)
                    msk_raw = io.imread(msk_path)
                    dsm_raw = io.imread(dsm_path)
            else:
                if self.inmemory:
                    img_raw, msk_raw = self.data[ii]
                else:
                    img_raw = io.imread(img_path)
                    msk_raw = io.imread(msk_path)

            toc = datetime.now()
            if self.verbose:
                print('getitem select image: %.2f',(toc-tic))
            tic = datetime.now()

            if len(img_raw.shape) == 2:
                img_raw = color.gray2rgb(img_raw)

            if self.use_dsm:
                img = np.full((img_raw.shape[0] + self.crop_size[0] - (img_raw.shape[0] % self.crop_size[0]),
                               img_raw.shape[1] + self.crop_size[1] - (img_raw.shape[1] % self.crop_size[1]),
                               img_raw.shape[2] + 1),
                              fill_value=0.0,
                              dtype=np.float32)
            else:
                img = np.full((img_raw.shape[0] + self.crop_size[0] - (img_raw.shape[0] % self.crop_size[0]),
                               img_raw.shape[1] + self.crop_size[1] - (img_raw.shape[1] % self.crop_size[1]),
                               img_raw.shape[2]),
                              fill_value=0.0,
                              dtype=np.float32)

            img[:img_raw.shape[0], :img_raw.shape[1], :img_raw.shape[2]] = img_raw
            if self.use_dsm:
                img[:dsm_raw.shape[0], :dsm_raw.shape[1], -1] = dsm_raw


            msk = np.full((msk_raw.shape[0] + self.crop_size[0] - (msk_raw.shape[0] % self.crop_size[0]),
                           msk_raw.shape[1] + self.crop_size[1] - (msk_raw.shape[1] % self.crop_size[1]),
                           msk_raw.shape[2]),
                          fill_value=0,
                          dtype=np.int64)
            msk[:msk_raw.shape[0], :msk_raw.shape[1]] = msk_raw

            toc = datetime.now()
            if self.verbose:
                print('getitem copy images: %.2f',(toc-tic))
            tic = datetime.now()

            msk = self.mask_to_class(msk)
            toc = datetime.now()
            if self.verbose:
                print('getitem mask_to_class: %.2f',(toc-tic))
            tic = datetime.now()
            msk, msk_true = self.shift_labels(msk)

            toc = datetime.now()
            if self.verbose:
                print('getitem shift: %.2f',(toc-tic))
            tic = datetime.now()

            # Normalization.
            if np.min(img)<0:
                img = img + np.min(img)
            else:
                img = img - np.min(img)
            img = (img / np.max(img)) - 0.5

            #print('local_batch_size: ',local_batch_size)

            img=np.array(img,dtype='float32')
            msk=np.array(msk,dtype='int16')
            msk_true=np.array(msk_true,dtype='int16')
            if self.mode in ['Train','ValidateTrain']:
                if self.select_non_match=='random':
                    local_random_crops = 50
                    if self.local_batch_size*40>50:
                        local_random_crops = self.local_batch_size*40

                    img, msk, msk_true = self.random_crops(img, msk, msk_true, local_random_crops)
                    toc = datetime.now()
                    if self.verbose:
                        print('getitem random1: %.2f',(toc-tic))
                    tic = datetime.now()
                    search_imgs = img[self.local_batch_size:,:,:,:]
                    search_msks = msk[self.local_batch_size:,:,:]
                    search_msks_true = msk_true[self.local_batch_size:,:,:]

                    img_m = np.copy(img[:self.local_batch_size,:,:,:])
                    msk_m = np.copy(msk[:self.local_batch_size,:,:])
                    msk_true_m =  np.copy(msk_true[:self.local_batch_size,:,:])

                    img_nm = np.copy(search_imgs[:self.local_batch_size,:,:,:])
                    msk_nm = np.copy(search_msks[:self.local_batch_size,:,:])
                    msk_true_nm =  np.copy(search_msks_true[:self.local_batch_size,:,:])

                    for ii in range(0, self.local_batch_size):
                        match = np.sum(msk_m[ii] == msk_m[ii])
                        for jj in range(0, search_imgs.shape[0]):
                            match_temp = np.sum(msk_m[ii] == search_msks[jj])
                            if match_temp<match:
                                img_nm[ii] = search_imgs[jj]
                                msk_nm[ii] = search_msks[jj]
                                msk_true_nm[ii] = search_msks_true[jj]
                                match = match_temp
                    toc = datetime.now()
                    if self.verbose:
                        print('getitem random2: %.2f',(toc-tic))
                    tic = datetime.now()
                else:
                    img, msk, msk_true = self.random_crops(img, msk, msk_true, self.local_batch_size) 
                    img_m = img
                    msk_m = msk
                    msk_true_m = msk_true

                    img_nm = img
                    msk_nm = np.copy(msk_m)
                    msk_true_nm = np.copy(msk_true_m)
                    max_class = self.num_classes - 1

                    if self.ignore_others_non_match:
                        max_class = self.num_classes - 2   

                    for b in range(0,msk_nm.shape[0]):
                        new_classes = np.random.permutation(max_class+1)

                        next_permutation = True
                        counter_p=0
                        while next_permutation:
                            next_permutation = False
                            counter_p+=1
                            new_classes = np.random.permutation(max_class+1)
                            for i in range(0,max_class + 1):
                                if i==new_classes[i]:
                                    next_permutation=True
                                    break

                        for i in range(0,max_class + 1):
                            msk_nm[b][msk_m[b]==i] = new_classes[i]
                        for i in range(0,max_class + 1):
                            msk_true_nm[b][msk_true_m[b]==i] = new_classes[i]

                #print("match % - ",str(100*match/(msk_nm.shape[1]*msk_nm.shape[2]*msk_nm.shape[0])))
                img_m = np.transpose(img_m, (0, 3, 1, 2))
                img_nm = np.transpose(img_nm, (0, 3, 1, 2))
                img_m = torch.from_numpy(img_m)
                msk_m = torch.from_numpy(msk_m)
                msk_true_m = torch.from_numpy(msk_true_m)
                img_nm = torch.from_numpy(img_nm)
                msk_nm = torch.from_numpy(msk_nm)
                msk_true_nm = torch.from_numpy(msk_true_nm)

                toc = datetime.now()
                if self.verbose:
                    print('getitem ending: %.2f',(toc-tic))
                tic = datetime.now()
                sys.stdout.flush()

                return img_m, msk_m, msk_true_m, img_nm, msk_nm, msk_true_nm 

            elif self.mode in ['Validate','Thresholds_Train','Test','Thresholds']:
                img, msk, msk_true = self.test_crops(img, msk, msk_true)

                img = np.transpose(img, (0, 1, 4, 2, 3))
                msk = np.transpose(msk, (0, 1, 2, 3))
                msk_true = np.transpose(msk_true, (0, 1, 2, 3))

            # Turning to tensors.
            img = torch.from_numpy(img).to(dtype=torch.float32)
            msk = torch.from_numpy(msk).to(dtype=torch.int64)
            msk_true = torch.from_numpy(msk_true).to(dtype=torch.int64)

            spl = img_path.split('/')
            # Returning to iterator.
            return img, msk, msk_true, spl[-1]

    def __len__(self):
        if self.dataset == 'GRSS':
            if self.mode == 'Train':
                return 1
            return 1
        else:
            if self.mode == 'Train':
                return self.dataset_size_factor*len(self.imgs)
            else:
                return len(self.imgs)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=5, help="number of epochs")
    
    parser.add_argument("-lr", "--lr", dest="lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("-momentum", "--momentum", dest="momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("-w", "--workers", dest="workers", type=int, default=10, help="data loarder workers")
    parser.add_argument("-w_size", "--w_size", dest="w_size",type=int, default=224, help="width for image resizing")
    parser.add_argument("-h_size", "--h_size", dest="h_size",type=int, default=224, help="height for image resizing")
    parser.add_argument("-n", "--n_classes", dest="n_classes",type=int, default=5, help="number of classes")
    parser.add_argument("-dataset", "--dataset", dest="dataset", default='Vaihingen', help="dataset name")
    parser.add_argument("-onehot", "--onehot", dest="onehot", type=str2bool, default=True, help="use one hot encoding for condition model")
    parser.add_argument("-select_non_match", "--select_non_match", dest="select_non_match", default='random', help="method for selecting non match patches (random,none)")
    parser.add_argument("-i", "--ignore_others_non_match", dest="ignore_others_non_match", default=False, help="ignore class -others- to generate non match patch")
    parser.add_argument("-cs", "--cs_name", dest="cs_name", default='unet', help="closed set model name")
    parser.add_argument("-save_images", "--save_images", dest="save_images", type=str2bool, default=True, help="save_images after generating metrics")
    parser.add_argument("-early_stop", "--early_stop", dest="early_stop",type=int, default=100, help="early stop training")
    parser.add_argument("-eval_type", "--eval_type", dest="eval_type", default='val_train', help="which images use to generate thresholds (val_train,train)")
    parser.add_argument("-ckpt_path", "--ckpt_path", dest="ckpt_path", default='/mnt/DADOS_GRENOBLE_1/ian/openseg/ckpt', help="base path to save trained models")
    parser.add_argument("-outp_path", "--outp_path", dest="outp_path", default='/mnt/DADOS_GRENOBLE_1/ian/openseg/outputs', help="base path to save metrics, logs, images and charts")
    parser.add_argument("-datasets_path", "--datasets_path", dest="datasets_path", default='/mnt/DADOS_GRENOBLE_1/ian/datasets', help="base path for dataset")
    
    parser.add_argument("-skip_connections", "--skip_connections", dest="skip_connections", default='csdec_all', help="list all skip connections to be used ('osenc,all,csenc,none')")
    parser.add_argument("-norm", "--normalizations", dest="normalizations", default='batch', help="list all kinds of normalizations to be used ('none,batch,instance')")
    parser.add_argument("-activations", "--activations", dest="activations", default='relu', help="list all skip activation to be used ('relu,leakyrelu')")
    parser.add_argument("-alphas", "--alphas", dest="alphas", default="0.95", help="alpha parameter to balance match and non match loss")
    
    parser.add_argument("-train", "--train", dest="train_model", type=str2bool, default=False, help="train model")
    parser.add_argument("-load", "--load", dest="load_model", type=str2bool, default=False, help="load pretreined model")
    parser.add_argument("-eval", "--eval", dest="eval_model", type=str2bool, default=False, help="generate all evaluation metrics")
    parser.add_argument("-o", "--onlynew", dest="only_new_model", type=str2bool, default=False, help="train only if the model do not exists")
    parser.add_argument("-prepeval", "--prepeval", dest="prep_eval", type=str2bool, default=False, help="train only if the model do not exists")
    parser.add_argument("-prepthresholds", "--prepthresholds", dest="prep_thresholds", type=str2bool, default=False, help="train only if the model do not exists")
    
    parser.add_argument("-metrics", "--metrics", dest="metrics", type=str2bool, default=True, help="calculate metrics if eval==true")
    
    parser.add_argument("-final_extra_convs_block", "--final_extra_convs_block", dest="final_extra_convs_block", default="0", help="how many convs the final extra conv block shoul have, 0 - means no extra block")
    parser.add_argument("-model", "--model", dest="model", default='OpenC2Seg', help="model name")
    parser.add_argument("-hidden", "--hidden", dest="hidden", default='0', help="hidden classes")
    parser.add_argument("-validaterate", "--validaterate", dest="validaterate", default=1, help="hidden classes")
    
    args = parser.parse_args()
    
    import sys
    IN_COLAB = 'google.colab' in sys.modules
    if IN_COLAB:
        args.datasets_path='datasets'
        args.ckpt_path='ckpt'
        args.outp_path='outputs'
        print("IN_COLAB")
    else:
        print("NOT_IN_COLAB")
    return args

def validate_train(loader, net, args, thresholds, overlap):
    #full_val_patches = get_predictions_reconstructions(loader, net, args)
    #full_imgs, full_msks, full_trues, full_prds, full_outs, full_minlosses = get_full_images(full_val_patches, overlap=overlap)
    
    full_imgs, full_msks, full_trues, full_prds, full_outs, full_minlosses = get_predictions_reconstructions(loader, net, args)

    flatten_full_msks  = get_flatten_image(full_msks)
    flatten_full_trues = get_flatten_image(full_trues)
    flatten_full_prds  = get_flatten_image(full_prds)
    flatten_full_minlosses = get_flatten_image(full_minlosses)
    
    return roc_auc_os(flatten_full_minlosses, flatten_full_trues, args.num_known_classes-1, thresholds), flatten_full_msks, flatten_full_trues, flatten_full_prds, flatten_full_minlosses

def loss_custom(input, target, mask_m=None, mask_nm=None, reduction='mean'):
    if mask_m is None or mask_nm is None:
        out = torch.abs(input-target)
    else:
        condition = mask_m == mask_nm
        new_condition = torch.zeros(input.shape, dtype=bool)

        for i in range(input.shape[1]):
            new_condition[:,i,:,:] = condition

        out = torch.abs(input[~new_condition]-target[~new_condition])

    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out.sum()

# Training procedure.
def train_os(train_loader, net, criterion, optimizer, epoch, num_known_classes, num_unknown_classes, hidden, train_args, train_log_path, alpha=0.9,print_intermediate_images=False, intermediate_images_path=''):
    net.train()
    
    for p in net.enc1.parameters():
        p.requires_grad = False
    for p in net.enc2.parameters():
        p.requires_grad = False
    for p in net.enc3.parameters():
        p.requires_grad = False
    for p in net.enc4.parameters():
        p.requires_grad = False
    for p in net.center.parameters():
        p.requires_grad = False
    for p in net.dec4.parameters():
        p.requires_grad = False
    for p in net.dec3.parameters():
        p.requires_grad = False
    for p in net.dec2.parameters():
        p.requires_grad = False
    for p in net.dec1.parameters():
        p.requires_grad = False
    for p in net.final.parameters():
        p.requires_grad = False
        
    if len(intermediate_images_path)==0 and print_intermediate_images:
        raise Exception("cannot print intermetiate images without a path")
        
    net.enc1.eval()
    net.enc2.eval()
    net.enc3.eval()
    net.enc4.eval()
    net.center.eval()
    net.dec4.eval()
    net.dec3.eval()
    net.dec2.eval()
    net.dec1.eval()    
    net.final.eval()
        
    tic = time.time()
    
    train_loss = []
    tic0 = time.time()
    for i, batch_data in enumerate(train_loader):
        toc0 = time.time()
        tic1 = time.time()
        
        inps_m, labs_m, true_m, inps_nm, labs_nm, true_nm = batch_data
        inps_m = inps_m.to(train_args.device)
        inps_nm = inps_nm.to(train_args.device)
        labs_m = labs_m.to(train_args.device)
        labs_nm = labs_nm.to(train_args.device)
        
        if (len(inps_m.shape)==5):
            inps_m.squeeze_(0)
            labs_m.squeeze_(0)
            inps_nm.squeeze_(0)
            labs_nm.squeeze_(0)
        
        optimizer.zero_grad()
        outs_m, outs_nm = net(inps_m.float(), (labs_m.float(), labs_nm.float()))
        
        match_loss = alpha*criterion(outs_m, inps_m)
        nonmatch_loss = (1-alpha)*criterion(inps_nm, outs_nm, labs_m, labs_nm)
        
        loss = match_loss + nonmatch_loss

        loss.backward()

        optimizer.step()
        
        train_loss.append((loss.data.item()))
    
    toc = time.time()
    
    train_loss = np.asarray(train_loss)
    
    print('-------------------------------------------------------------------')
    print('[epoch %d], [train rec loss %.4f +/- %.4f], [training time %.2f]' % (
        epoch, train_loss[:].mean(), train_loss[:].std(), (toc - tic)))
    #print('-------------------------------------------------------------------')
    return train_loss


def config_execution(args,dataset, hidden, task, epochs=None):
    args.final_extra_convs_block = '0'
    
    if task=='train' or task=='all':
        args.train_model=True
    if task=='prepare_evaluation' or task=='all':
        args.prep_eval=True
    if task=='define_thresholds' or task=='all':
        args.prep_thresholds=True
    if task=='eval' or task=='all':
        args.eval_model=True

    if dataset=='vaihingen':
        args.dataset='Vaihingen'
        if hidden==0:
            args.alphas='0.92'
        elif hidden==1:
            args.alphas='0.89'
        elif hidden==2:
            args.alphas='0.89'
        elif hidden==3:
            args.alphas='0.93'
        elif hidden==4:
            args.alphas='0.94'
    elif dataset=='potsdam':
        args.dataset='Potsdam'
        if hidden==0:
            args.alphas='0.95'
        elif hidden==1:
            args.alphas='0.85'
        elif hidden==2:
            args.alphas='0.94'
        elif hidden==3:
            args.alphas='0.92'
        elif hidden==4:
            args.alphas='0.85'
    args.epochs=100
    if epochs is not None:
        args.epochs=int(epochs)
            

    args.hidden=str(hidden)
    args.backgroundclasses = []
    hidden = []
    if '_' in args.hidden:
        hidden = [int(h) for h in args.hidden.split('_')]
    else:
        hidden = [int(args.hidden)]
        
    args.hidden = hidden
    args.hidden_classes = hidden
    args.num_known_classes = args.n_classes - len(hidden)
    args.num_unknown_classes = len(hidden)
    
    now = datetime.now() # current date and time
    datestr = str(now.strftime("%Y%m%d%H%M%S"))
    
    # Setting experiment name.
    exp_name = args.model + '_' + args.cs_name+ "_"+ args.dataset + "_" + str(args.hidden_classes[0])

    pretreined_path_closedset = args.ckpt_path+'/'+args.cs_name+'_'+args.dataset+'_base_dsm_'+str(args.hidden_classes[0])+'/model_600.pth'
    
    if args.dataset=='Vaihingen':
        pretreined_path_closedset = args.ckpt_path+'/'+args.cs_name+'_'+args.dataset+'_base_dsm_'+str(args.hidden_classes[0])+'/model_1200.pth'

    print("ARQUIVO PRETREINO EXISTE? ",os.path.isfile(pretreined_path_closedset))

    pretrained_path = os.path.join(args.ckpt_path, exp_name, 'model_os_' + str(args.select_non_match)+"_" + str(args.ignore_others_non_match) + "_" + str(args.epochs) + '.pth')

    check_mkdir(os.path.join(args.ckpt_path))
    check_mkdir(os.path.join(args.ckpt_path, exp_name))
    check_mkdir(os.path.join(args.outp_path))

    check_mkdir(os.path.join(args.outp_path,'images'))
    check_mkdir(os.path.join(args.outp_path,'charts'))
    check_mkdir(os.path.join(args.outp_path,'images',exp_name))
    check_mkdir(os.path.join(args.outp_path,'charts',exp_name))
    check_mkdir(os.path.join(args.outp_path,'charts',exp_name,'roc'))
    check_mkdir(os.path.join(args.outp_path,'charts',exp_name,'trainlog'))

    check_mkdir(os.path.join(args.outp_path,exp_name))
    final_outp_path = os.path.join(args.outp_path,exp_name)

    check_mkdir(os.path.join(final_outp_path,'images'))
    check_mkdir(os.path.join(final_outp_path,'charts'))
    check_mkdir(os.path.join(final_outp_path,'charts','roc'))
    check_mkdir(os.path.join(final_outp_path,'charts','trainlog'))

    images_path = os.path.join(args.outp_path,'images',exp_name)
    charts_path = os.path.join(args.outp_path,'charts',exp_name)

    images_path_roc = os.path.join(charts_path,'roc')
    images_path_trainglog = os.path.join(charts_path,'trainlog')
    metrics_path = os.path.join(final_outp_path, 'metrics_'+args.eval_type+"_"+datestr+'.csv')

    # Setting device [0|1|2].
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(device)
    
    args.thresholds = [ i/100 for i in range(90,101)]
    args.thresholds.append(0.85)
    args.thresholds.append(0.80)
    args.thresholds.append(0.75)
    args.thresholds.append(0.70)
    args.thresholds.sort()
    
    if args.dataset=='Vaihingen':
        args.local_batch_size=2
    else:
        args.local_batch_size=1

    
    return args, exp_name, pretrained_path, datestr, final_outp_path, images_path_roc, images_path_trainglog, metrics_path, pretreined_path_closedset, images_path, charts_path

def get_predictions_reconstructions(test_loader, net, args, print_intermediate_images=False, intermediate_images_path=''):           
    # Setting network for evaluation mode.
    num_known_classes = int(args.n_classes)-len(str(args.hidden_classes).split('_'))

    net.eval()

    img_list = []
    msk_list = []
    dsm_list = []
    true_list = []
    outs_list = []
    minlosses_list = []
    prds_list = []

    with torch.no_grad():    
        tic = time.time()
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            print('get prd Batch %d/%d' % (i + 1, len(test_loader)))
            sys.stdout.flush()
            # Obtaining images, labels and paths for batch.
            inps_batch, labs_batch, true_batch, img_name = data                
                
            inps_batch = inps_batch.squeeze()
            labs_batch = labs_batch.squeeze()
            true_batch = true_batch.squeeze()
            
            #print(inps_batch.shape)
            img_full = []
            msk_full = []
            true_full = []
            dsm_full = []
            outs_full = []
            minlosses_full = []
            # Iterating over patches inside batch.
            #tic = time.time()
            prds_full = []

            for j in range(inps_batch.size(0)):
                print('    get prd MiniBatch %d/%d' % (j + 1, inps_batch.size(0)))
                sys.stdout.flush()
                
                img_row_list = []
                msk_row_list = []
                true_row_list = []
                dsm_row_list = []
                outs_row_list = []
                minlosses_row_list = []
                for k in range(inps_batch.size(1)):
                    inps = inps_batch[j, k].unsqueeze(0)
                    labs = labs_batch[j, k].unsqueeze(0).unsqueeze(0)
                    true = true_batch[j, k].unsqueeze(0)
                    # Casting to cuda variables.
                    imgs = inps.cuda()#args['device'])
                    msks = labs.cuda()#args['device'])
                    true = true.cuda()#args['device'])
                    
                    prds = None
                    rec_losses = []
                    outs, enc1, enc2, enc3, enc4, center, dec1, dec2, dec3, dec4  = net.segment_forward(imgs)
                    # semantic segmentation prediction
                    # Computing probabilities.
                    soft_outs = F.softmax(outs, dim=1)
                    # Obtaining prior predictions.                    
                    prds = soft_outs.data.max(1)[1]
                    #print(outs.shape)
                    prds_full.append(prds.cpu().numpy())
                    
                    img_unique_seq=str(i)+'_'+str(j)+'_'+str(k)
                    
                    if print_intermediate_images:
                        visible = imgs.detach().cpu().squeeze().permute((1,2,0))[:,:,0:3].numpy()                        
                        visible = visible - np.min(visible)
                        visible = visible/np.max(visible)
                        plt.imsave(os.path.join(intermediate_images_path,'test_'+img_unique_seq+'_img.png'), visible, format='png', dpi=300)
                        plt.imsave(os.path.join(intermediate_images_path,'test_'+img_unique_seq+'_preds.png'), prds.detach().cpu().squeeze().numpy(), format='png', dpi=300)
                    
                    for ii in range(num_known_classes):
                        condition = np.ones(msks.shape)*ii
                        condition = torch.from_numpy(condition).cuda().float()
                        # Forward
                        recs = net.condition_forward(condition, enc1, enc2, enc3, enc4, center, dec1, dec2, dec3, dec4 )
                        # Reconstruction prediction
                        rec_loss = torch.abs(recs-imgs).mean(axis=1)
                        rec_losses.append(rec_loss.cpu().numpy())
                        
                        if print_intermediate_images:
                            visible = recs.detach().cpu().squeeze().permute((1,2,0))[:,:,0:3].numpy() 
                            visible[:,:,0] = visible[:,:,0] - np.min(visible[:,:,0])
                            visible[:,:,0] = visible[:,:,0]/np.max(visible[:,:,0])
                            visible[:,:,1] = visible[:,:,1] - np.min(visible[:,:,1])
                            visible[:,:,1] = visible[:,:,1]/np.max(visible[:,:,1])
                            visible[:,:,2] = visible[:,:,2] - np.min(visible[:,:,2])
                            visible[:,:,2] = visible[:,:,2]/np.max(visible[:,:,2])
                            plt.imsave(os.path.join(intermediate_images_path,'test_'+img_unique_seq+'_cond_'+str(ii)+'.png'), visible, format='png', dpi=300)
                        
                    rec_losses = np.array(rec_losses)
                    min_loss = np.min(rec_losses, axis=0).squeeze()
                    
                    img_np = util.img_as_ubyte(np.transpose(((imgs.cpu().squeeze().numpy()[0:3,:,:] + 0.5) * 255).astype(np.uint8), (1, 2, 0)))
                    dsm_np = util.img_as_ubyte(((imgs.cpu().squeeze().numpy()[-1,:,:] + 0.5) * 255).astype(np.uint8))
                    true_np = util.img_as_ubyte(true.cpu().squeeze().numpy().astype(np.uint8))
                    msk_np = util.img_as_ubyte(msks.cpu().squeeze().numpy().astype(np.uint8))
                    outs_np = outs.cpu().numpy()
                    
                    img_row_list.append(img_np)
                    msk_row_list.append(msk_np)
                    dsm_row_list.append(dsm_np)
                    true_row_list.append(true_np)
                    outs_row_list.append(outs_np)
                    minlosses_row_list.append(min_loss)
                
                img_full.append(img_row_list)
                msk_full.append(msk_row_list)
                true_full.append(true_row_list)
                outs_full.append(outs_row_list) 
                minlosses_full.append(minlosses_row_list)
            
            full_imgs, full_msks, full_trues, full_prds, full_outs, full_minlosses = get_full_images([[img_full],[msk_full],[true_full],[outs_full],[minlosses_full]], overlap=test_loader.dataset.overlap)
            
            img_list.append(full_imgs[0])
            msk_list.append(full_msks[0])
            true_list.append(full_trues[0])
            outs_list.append(full_outs[0])
            outs_full = np.array(full_outs[0])
            minlosses_list.append(full_minlosses[0])
            prds_list.append(full_prds[0])

        toc = time.time()
        print('        Elapsed Time: %.2f' % (toc - tic))

    sys.stdout.flush()
            
    return img_list, msk_list, true_list, prds_list, outs_list, minlosses_list

def get_full_images(full_patches, overlap=True):
    imgs = full_patches[0]
    msks = full_patches[1]
    trues = full_patches[2]
    outs = full_patches[3]
    minlosses = full_patches[4]
    if len(full_patches)==6:
        dsms = full_patches[5]

    stride = (224 // 2)
    if overlap==False:
        stride = 224
    
    full_imgs  = []
    full_msks  = []
    full_prds  = []
    full_trues = []
    full_outs = []
    full_minlosses = []
    for i in range(len(imgs)):    
        image_shape = (stride*(len(imgs[i]) + 1),stride*(len(imgs[i][0]) + 1),3)
        msk_shape   = (stride*(len(imgs[i]) + 1),stride*(len(imgs[i][0]) + 1))
        outs_shape  = (1, outs[i][0][0].shape[1], stride*(len(imgs[i]) + 1),stride*(len(imgs[i][0]) + 1))

        full_img  = np.zeros(image_shape, dtype=np.uint8)
        full_msk  = np.zeros(msk_shape, dtype=np.uint8)
        #full_prd  = np.zeros(msk_shape, dtype=np.uint8)
        full_true = np.zeros(msk_shape, dtype=np.uint8)
        full_out = np.zeros(outs_shape)
        full_minloss = np.zeros(msk_shape)
        how_many_added_pixels = np.zeros(msk_shape)

        for y in range(len(imgs[i])):          
            for x in range(len(imgs[i][y])):
                #print(i,y,x)
                img = imgs[i][y][x]
                msk = msks[i][y][x]
                prd = msks[i][y][x]
                true = trues[i][y][x]
                out  = outs[i][y][x]
                minloss  = minlosses[i][y][x]

                # Computing Priors.
                offset_y = y*stride
                offset_x = x*stride
                full_img[offset_y:offset_y+224, offset_x:offset_x+224, :] = img[:,:,0:4]
                full_msk[offset_y:offset_y+224, offset_x:offset_x+224] = msk
                full_true[offset_y:offset_y+224, offset_x:offset_x+224] = true

                how_many_added_pixels[offset_y:offset_y+224, offset_x:offset_x+224] += 1

                full_out[:,:,offset_y:offset_y+224, offset_x:offset_x+224] += out
                full_minloss[offset_y:offset_y+224, offset_x:offset_x+224] += minloss
                
        full_out = full_out/how_many_added_pixels
        full_out = torch.from_numpy(full_out)
        soft_outs = F.softmax(full_out, dim=1)
        # Obtaining prior predictions.                    
        prds = soft_outs.data.max(1)[1]
        del soft_outs
        prds = np.array(prds.squeeze().numpy(),dtype=np.uint8)
        #print('2', np.unique(prds,return_counts=True))

        y0, y1, x0, x1=trim_coords(np.sum(full_img,axis=2))
        full_img = full_img[y0:y1,x0:x1]
        full_msk = full_msk[y0:y1,x0:x1]
        full_out = full_out[:,:,y0:y1,x0:x1]
        full_true = full_true[y0:y1,x0:x1]
        full_minloss = full_minloss/how_many_added_pixels
        
        full_minloss = full_minloss[y0:y1,x0:x1]
        prds = prds[y0:y1,x0:x1]
        
        full_imgs.append(full_img)
        full_msks.append(full_msk)
        full_prds.append(prds)
        full_trues.append(full_true)
        full_outs.append(full_out)
        full_minlosses.append(full_minloss)
        #show_minloss = full_minloss - np.min(full_minloss)
        #show_minloss = show_minloss/np.max(show_minloss)
        #show_images([[full_img,full_msk,full_true,prds,show_minloss]])
    return full_imgs, full_msks, full_trues, full_prds, full_outs, full_minlosses

def trim_coords(img):

    # Mask of non-black pixels (assuming image has a single channel).
    bin = img != 0

    # Coordinates of non-black pixels.
    coords = np.argwhere(bin)

    # Bounding box of non-black pixels.
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    return y0, y1, x0, x1

def get_flatten_image(img):
    assert len(img)>0 

    
    totalpixels=0
    #obj_type = None
    for i in range(0,len(img)):
    #    obj_type = img[i].dtype
        totalpixels+=img[i].shape[0]*img[i].shape[1]

    #assert obj_type is not None
    #print(obj_type)
    
    flatten_img = np.zeros(totalpixels)#, dtype = obj_type)
    
    totalpixels=0
    for i in range(0,len(img)):
        f = img[i].flatten()
        flatten_img[totalpixels:totalpixels+f.shape[0]] = f

        totalpixels += img[i].shape[0]*img[i].shape[1]
    return flatten_img


def get_quantiles(flatten_full_minlosses, flatten_full_msks, flatten_full_prds, thresholds):
    losses_by_class=[]
    print(np.unique(flatten_full_msks,return_counts=True))
    print(np.unique(flatten_full_prds,return_counts=True))    
    for c in range(0,int(np.max(flatten_full_msks))+1):
        print('Pixels by class %d...' % (c))
        sys.stdout.flush()

        losses_list_by_class = flatten_full_minlosses[(flatten_full_msks == c) & (flatten_full_prds == c)]

        print('Pixels by class:', str(losses_list_by_class.shape),' - ', np.sum(flatten_full_msks == c))
        losses_by_class.append(losses_list_by_class.flatten())

    losses_by_class = np.asarray(losses_by_class)
    losses_by_class = np.concatenate(losses_by_class)

    thresholds_values = np.quantile(losses_by_class, thresholds).tolist()
    return thresholds_values

def get_os_metrics2(net_unique_name, args, flatten_full_minlosses, flatten_full_msks, flatten_full_prds, flatten_superpixels, num_known_classes, thresholds, thresholds_values, metrics_path, image_path):
    
    tic = time.time()
    roc_auc_metrics = roc_auc_os(flatten_full_minlosses, flatten_full_msks, num_known_classes, thresholds)
    save_roc_fig(roc_auc_metrics,image_path)
    print('        ROC AUC Elapsed Time: %.2f' % (time.time() - tic))
    
    if args.metrics:
        print('generating metrics')
        restuls = Parallel(n_jobs=10)(delayed(parallel_metrics)(t,net_unique_name, args, flatten_full_minlosses, flatten_full_msks, flatten_full_prds, flatten_superpixels, num_known_classes, thresholds, thresholds_values, metrics_path, image_path,roc_auc_metrics) for t in range(0,len(thresholds)))
        print(restuls)
    return roc_auc_metrics

def roc_auc_os(minloss, msk, num_known_classes=4, thresholds=[0.7,0.99,1.0]):
    
    minloss_temp = minloss.ravel()
    msk_temp = msk.ravel()
    minloss_temp = minloss_temp[msk_temp <= num_known_classes]
    msk_temp = msk_temp[msk_temp <= num_known_classes]
    
    bin_msk_temp = (msk_temp == num_known_classes)

    # Computing ROC and AUC.
    print('    Computing AUC ROC...')
    fpr_c2ae, tpr_c2ae, t_c2ae = metrics.roc_curve(bin_msk_temp, minloss_temp)
    
    ind_c2ae = []
    for t in thresholds:
        if t == 1.00:
            ind_c2ae.append(-1)
        else:
            for i in range(len(t_c2ae)):
                if tpr_c2ae[i] >= t:
                    ind_c2ae.append(i)
                    break
    
    #print('    Computing c2ae AUC...')
    auc_c2ae = metrics.roc_auc_score(bin_msk_temp, minloss_temp)
    print('    AUC: ',auc_c2ae)
    return t_c2ae, tpr_c2ae, fpr_c2ae, ind_c2ae, auc_c2ae

class logloss():
    def __init__(self, filepath, savefilepath='', prefix='', savefile=False):
        self.train_loss = []
        self.validation_loss = []
        self.epochs=[]
        self.name = ""
        self.savefilepath=savefilepath
        self.start_plotting_epoch=5
        self.savefile=savefile
        print("log file fath:",filepath)
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                splitted_line = line.split(';')
                self.train_loss.append(float(splitted_line[14]))
                self.validation_loss.append(float(splitted_line[17]))
                self.epochs.append(int(splitted_line[13]))
            self.name=prefix+splitted_line[1]+"_"+splitted_line[3]+"_"+splitted_line[13]+"_"+splitted_line[6]+"_"+splitted_line[7]+"_"+splitted_line[8]+"_"+splitted_line[9]+"_"+splitted_line[10]+"_"+splitted_line[11]+"_"+splitted_line[12]
            #print(splitted_line)
            self.savename=self.name.strip(' ').replace("-","_").replace(' ','').replace('.','')
            self.train_loss_moving = self.moving_average(self.train_loss)
            self.validation_loss_moving = self.moving_average(self.validation_loss)
    def plot_chart(self):
        if len(self.validation_loss[self.start_plotting_epoch:])<=0:
            return
        plt.figure(figsize=(16,6))
        plt.title(self.name)
        plt.plot(self.validation_loss[self.start_plotting_epoch:],label="validation")
        plt.plot(self.train_loss[self.start_plotting_epoch:],label="train")
        plt.plot(self.validation_loss_moving[self.start_plotting_epoch:],label="validation MA")
        plt.plot(self.train_loss_moving[self.start_plotting_epoch:],label="train MA")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        label_x = [a for a in self.epochs[self.start_plotting_epoch:] if a % 10 == 0 ]
        label_x_values = np.array(label_x)-self.start_plotting_epoch
        plt.xticks(label_x_values,label_x)#self.epochs[10:],[])
        plt.legend()
        
        if self.savefile:
            print(self.savefilepath)
            plt.savefig(self.savefilepath, dpi=80)
            plt.close()
        else:
            plt.show()
            plt.close()
    def moving_average(self, values_list):
        i = 0
        moving_averages = []
        window_size=5
        while i < len(values_list):
            if i-window_size<0:
                moving_averages.append(0)
            else:
                this_window = values_list[i - window_size: i ]
                window_average = sum(this_window) / window_size
                moving_averages.append(window_average)
            i += 1
        return moving_averages
    
import matplotlib

def save_images(imgs, images_path, name, norm=False):
    
    for i in range(0,len(imgs)):
        img =  np.copy(imgs[i])
        print(os.path.join(images_path,name+"_"+str(i)+".npz"))
        np.savez(os.path.join(images_path,name+"_"+str(i)+".npz"), img=img)
        if norm:
            img = img - np.min(img)
            img = img / np.max(img)
        matplotlib.image.imsave(os.path.join(images_path,name+"_"+str(i)+".jpg"), img)
        
import time
def load_images_array(images_path, save_name, suffix, qtd_imgs):
    
    msks = []
    trues = []
    prds = []
    minlosses = []
    #print(type(qtd_imgs))
    if str(type(qtd_imgs)) in ["<class 'list'>","<class 'torch.utils.data.dataloader.DataLoader'>"]:
        qtd_imgs = len(qtd_imgs)
    
    for i in range(0,qtd_imgs):

        while os.path.isfile(os.path.join(images_path,save_name+"_full_minloss"+suffix+"_"+str(i)+".npz"))==False:
            time.sleep(120) 
            print("waiting: ",os.path.join(images_path,save_name+"_full_minloss"+suffix+"_"+str(i)+".npz"))
        
        msks.append(np.load(os.path.join(images_path,save_name+"_full_msk"+suffix+"_"+str(i)+".npz"))['img'])
        trues.append(np.load(os.path.join(images_path,save_name+"_full_true"+suffix+"_"+str(i)+".npz"))['img'])
        prds.append(np.load(os.path.join(images_path,save_name+"_full_prd"+suffix+"_"+str(i)+".npz"))['img'])
        minlosses.append(np.load(os.path.join(images_path,save_name+"_full_minloss"+suffix+"_"+str(i)+".npz"))['img'])

    return msks,trues,prds,minlosses

def save_roc_fig(roc_auc_metrics, image_path):
    
    print(image_path)
    print(roc_auc_metrics)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)

    lw = 2

    ax.plot(roc_auc_metrics[2], roc_auc_metrics[1], color='crimson', lw=lw, label='AUC C2AE: %0.3f' % roc_auc_metrics[-1])

    ax.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    ax.set_xlabel('FPR', size=25)
    ax.set_ylabel('TPR', size=25)

    ax.legend(loc='lower right', prop={'size': 25})

    plt.tight_layout()

    plt.savefig(image_path, dpi=80)
    plt.close()
    
from skimage import color

def parallel_metrics(t,net_unique_name, args, flatten_full_minlosses, flatten_full_msks, flatten_full_prds, flatten_superpixels, num_known_classes, thresholds, thresholds_values, metrics_path, image_path, roc_auc_metrics):
    print (t)
    os_prds = None
    os_prds = np.copy(flatten_full_prds)
    unknown_pixels = flatten_full_minlosses >= thresholds_values[t]
    os_prds[unknown_pixels] = num_known_classes

    total_metrics = get_curr_metric(flatten_full_msks,os_prds,num_known_classes)

    str_log = 'full_image;'+net_unique_name+';'
    str_log += str(thresholds[t])+';'+str(thresholds_values[t])+';'
    str_log += str(args.epochs)+';'+str(args.lr)+';'+str(args.weight_decay)+';'+str(args.momentum)+';'
    str_log += str(args.n_classes)+';'+str(args.select_non_match)+';'+str(args.ignore_others_non_match)+';'
    str_log += str(args.hidden_classes)+';'+str(args.dataset)+';'+str(num_known_classes)+';'+str(args.num_unknown_classes)+';'
    str_log += str(total_metrics[0])+';'
    str_log += str(total_metrics[1])+';'
    str_log += str(total_metrics[2])+';'
    str_log += str(total_metrics[3])+';'
    str_log += str(total_metrics[4])+';'
    str_log += str(total_metrics[5])+';'
    str_log += str(roc_auc_metrics[4])+';\n'

    f = open(metrics_path, "a")
    f.write(str_log)
    f.close()
    return t

def get_curr_metric(msk, prd, n_known):
    
    tru_np = msk.ravel()
    prd_np = prd.ravel()
    
    #print(np.unique(tru_np,return_counts=True))
    #print(np.unique(prd_np,return_counts=True))
    
    tru_valid = tru_np[tru_np <= n_known]
    prd_valid = prd_np[tru_np <= n_known]
    
    #print('        Computing CM...')
    cm = metrics.confusion_matrix(tru_valid, prd_valid)
    
    #print(cm)

    #print('        Computing Accs...')
    tru_known = 0.0
    sum_known = 0.0

    for c in range(n_known):
        tru_known += float(cm[c, c])
        sum_known += float(cm[c, :].sum())

    acc_known = float(tru_known) / float(sum_known)

    tru_unknown = float(cm[n_known, n_known])
    sum_unknown_real = float(cm[n_known, :].sum())
    sum_unknown_pred = float(cm[:, n_known].sum())
    
    pre_unknown = 0.0
    rec_unknown = 0.0
    
    if sum_unknown_pred != 0.0:
        pre_unknown = float(tru_unknown) / float(sum_unknown_pred)
    if sum_unknown_real != 0.0:
        rec_unknown = float(tru_unknown) / float(sum_unknown_real)
        
    acc_unknown = (tru_known + tru_unknown) / (sum_known + sum_unknown_real)
    
    acc_mean = (acc_known + acc_unknown) / 2.0
    
    #print('        Computing Balanced Acc...')
    bal = metrics.balanced_accuracy_score(tru_valid, prd_valid)
    
    #print('        Computing Kappa...')
    kap = metrics.cohen_kappa_score(tru_valid, prd_valid)
    
    #print('        Computing AUC...')
    #auc = metrics.auc(prd_valid, tru_valid)
    
    curr_metrics = [acc_known, acc_unknown, pre_unknown, rec_unknown, bal, kap]
    
    return curr_metrics