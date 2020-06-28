#!/home/sci/samlev/bin/bin/python3

#SBATCH --time=21-00:00:00 # walltime, abbreviated by -t
#SBATCH --mem=140G
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)        
#SBATCH -e slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --gres=gpu:5

import argparse
import os
from six.moves import range

#from scipy.misc import imsave
import pickle as pickle
#import tensorflow as tf
#import tensorflow as tf #.compat.v1 as tf
#tf.disable_v2_behavior()
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.utils import get_rng
from tensorpack.dataflow.base import RNGDataFlow
import numpy as np
from PIL import Image
import cv2
import copy
#from memory_profiler import profile

from LyCNN.datapack.IO import VisusDataflow
from LyCNN.datapack.medical_aug import normalize_staining, hematoxylin_eosin_aug



#sys.path.append(os.getcwd())
prediction_dir = "/home/sci/samlev/convNet_Lymphoma_Classifier/data/Unknowns/predictions/"

"""
Data flow similar to cifar10 http://tensorpack.readthedocs.io/en/latest/_modules/tensorpack/dataflow/dataset/cifar.html 
structure explained here: https://www.cs.toronto.edu/~kriz/cifar.html
Summary: Data stream capable of parallel feed with images 32x32x32x3 (RGB) 

 data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
 labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

Example Call: python denseNet_Lymphoma.py --tot train --drop_1 2 --drop_2 4 --depth 26 --max_epoch 1

"""
val_or_test = [['blackbox', False], ['test', True], ['dlbcl_test', False]][1]

def save_original(img, save_dir = None, name="temp"):
    img = np.transpose(img, [1, 2, 0])
    im = Image.fromarray(np.asarray(img).astype('uint8'),'RGB')
    # create dir to save originals
    if not os.path.exists('../data/Originals/'):
        os.mkdirs('../data/Originals/')
    im.save('../data/Originals/'+save_dir+'/'+str(name)+'.jpeg')
    im.close()

def read_data(filename):
    with open(filename, 'rb') as fo:
        try:
            dic = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x                                                                                 
            fo.seek(0)
            dic = pickle.load(fo, encoding='latin1')
        
        data = dic['data']
        label = dic['labels']
        fo.close()
    return (data, label)

def read_lymphoma(filenames,  train_or_test = 'train', image_size = 448, scale_size = 224, scale = 2, multi_crop = 0, crop_per_case = None, normalize = None, original_dir=None):
    num_show=5
    ret = []
    class_0 = 0
    class_1 = 0
    total_crops = 0
    unique_samples = 0
    for fname in filenames:
        
        with open(fname, 'rb') as fo:
            try:
                dic = pickle.load(fo)
            except UnicodeDecodeError:  #python 3.x
                fo.seek(0)
                dic = pickle.load(fo, encoding='latin1')
                
        data = dic['data']
        label = dic['labels']
        fo.close()
        
        crop_count = 0
        unique_samples += len(data)
        
        if crop_per_case is not None:
            crop_per_case = min(len(data), crop_per_case)
        else:
            crop_per_case = 1
        
        if bool(multi_crop) == False:
            multi_crop = 0
        
        print(">>>>>> will take ", str(crop_per_case)," images from each case for ",str(multi_crop), " crops.")
        print(">>>>>> Totaling crops: ", str(crop_per_case*multi_crop))
        
        for k in range(len(data)):#part):
            # resize to divisable format for convolutions
            img = data[k].astype("uint8")
            cropsize = (np.max(np.array(img.shape)), np.max(np.array(img.shape)[np.array(img.shape) < np.max(np.array(img.shape))]), 3)
            if scale_size is None:
                scaleSize = image_size,image_size #224*scale,224*scale
                imSize = image_size#image_size*scale #224
            else:
                scaleSize = image_size,image_size#scale_size,scale_size                             
                imSize = image_size
            #randPos = rnd.choice([0, 50, 100, 200, 300])
            #img = data[k][:, randPos:(randPos+imSize), randPos:(randPos+imSize)] #:32, :32] #size currently (927,1276,3)
            
            if original_dir:
                if not os.path.exists(original_dir):
                    os.mkdirs(original_dir)
                save_original(data[k][:,:, :], save_dir=original_dir, name=k)
            
            ##quality_cropper = quality_random_crop(data[k][:,:, :],imSize)
            ##img = quality_cropper.random_crop_select()  
            
            # make rgb feasible
            img = np.transpose(img, [1, 2, 0])
            img_og = copy.deepcopy(img)
            
            
            aspect_ratio = min( img.shape[1]/float(img.shape[0]) , img.shape[0]/float(img.shape[1]) )
            acc_scaleSize = (int(scaleSize[0]*aspect_ratio),scaleSize[1]) if img.shape[0] < img.shape[1] else (scaleSize[0],int(aspect_ratio*scaleSize[1]))
            W = int( imSize*img.shape[1]/float(img.shape[0]) )
            H = int( imSize*img.shape[0]/float(img.shape[1]) )
            
            #print(" True size after scaling to preserve aspect ratio: ", scaleSize)
            #print("larger crop size: ", W," ",H)
            
            multi_crop_ = 1
            if crop_count < crop_per_case:
                multi_crop_ = multi_crop+1
                crop_count += 1
            for tile in range(multi_crop_):
                if multi_crop_ != 1:
                    total_crops += 1
                start_w = [100, 300, 500, 700][tile] if multi_crop_ <= 4 else [100, 200, 300, 400, 500, 600][tile]
                start_h = [400, 400, 400, 400][tile] if multi_crop_ <= 4 else [400, 400, 400, 400, 400, 400][tile]
                
                copy_func = copy.deepcopy if multi_crop_ != 1 else lambda x: x
                img_crop = copy.deepcopy(img_og)
                
                img_crop = img_crop[start_w:(start_w+2*image_size),start_h:(start_h+2*image_size),:]
                
                #img_stack = np.expand_dims( img_crop, axis=0)
                
                img_crop = Image.fromarray(img_crop,'RGB')

                img_crop = img_crop.resize((image_size,image_size), Image.ANTIALIAS)
                img_crop = np.array(img_crop)
                img_crop= img_crop.reshape((image_size,image_size,3))#.transpose(0,3,1,2)
                
                # to place as nd array for tensorflow and augmentors
                if label[k] == 0:
                    #label[k] = 0.1
                    class_0 += 1
                elif label[k] == 1:
                    #label[k] = 0.9
                    class_1 += 1

                if label[k] != 0 and label[k] != 1:
                    for i in range(20):
                        print(" >>>>>>>>>>>>>> LABELING INCORRECT> VALUE: ",label[k])

                if normalize:
                    img_crop = normalize_staining().apply_image(img_crop)
                
                ret.append([img_crop.astype("uint8"), label[k]])
                #img = copy.deepcopy(img_og)
                
    print(">>>> Total crops observed: ", total_crops)
    return (ret, class_0, class_1, unique_samples)

# read data and write to idx file with z space
# filling order for controlable resolution
"""memory_log = "no_name.log"
memory_profile_log=open(memory_log,'w+')
@profile(stream=memory_profile_log)"""
def read_write_idx_lymphoma(filenames, train_or_test='train', image_size=448, scale_size=224, scale=2, multi_crop=0
                            ,crop_per_case=None, normalize=None, original_dir=None
                            , write_crop=True, idx_filepath='', mode=None, resolution=None):
    num_show = 5
    ret = []
    class_0 = 0
    class_1 = 0
    total_crops = 0
    unique_samples = 0
    idx_samples = []
    idx_labels = []
    data=[]
    for fname in filenames:
        if mode == 'w':
            with open(fname, 'rb') as fo:
                try:
                    dic = pickle.load(fo)
                except UnicodeDecodeError:  # python 3.x
                    fo.seek(0)
                    dic = pickle.load(fo, encoding='latin1')
            data = dic['data']
            label = dic['labels']
            fo.close()

        if mode == 'r':
            idx_sample = VisusDataflow.ReadData(data=fname, load=True, resolution=resolution)
            idx_samples.append(idx_sample.data)
            idx_labels.append(int(str(1) in fname.split('~')[-1]))
            #continue collecting from folder
            if fname != filenames[-1]:
                continue
            data = idx_samples
            label = idx_labels

        crop_count = 0
        unique_samples += len(data)

        if crop_per_case is not None:
            crop_per_case = min(len(data), crop_per_case)
        else:
            crop_per_case = 1

        if bool(multi_crop) == False:
            multi_crop = 0

        print(">>>>>> will take ", str(crop_per_case), " images from each case for ", str(multi_crop), " crops.")
        print(">>>>>> Totaling crops: ", str(crop_per_case * multi_crop))

        # class for writing images into IDX format
        # with z space filling order for parameter based
        # resolution on read
        if mode == 'w':
            visusWriter = VisusDataflow.WriteZOrder()
            if not os.path.exists(idx_filepath):
                os.mkdir(idx_filepath)

        for k in range(len(data)):  # part):
            # resize to divisable format for convolutions
            img = data[k].astype("uint8")
            # make rgb feasible
            if img.shape[0] == 3:
                img = np.transpose(img, [1, 2, 0])
            #if max(img.shape) != img.shape[0]:
            #    img = np.transpose(img, [2, 1, 0])
            img_og = copy.deepcopy(img)


            if mode == 'w':
                s = ""
                idx_filename = fname.split('/')[-1].split('.')[0] + '_' + str(k) + '~' + str(label[k]) + '.idx'
                #print(os.path.join(idx_filepath, idx_fname))
                visusWriter.convert_image(image=copy.deepcopy(img_og),idx_filename=os.path.join(idx_filepath,idx_filename))



            if mode == 'r':
                cropsize = (
                np.max(np.array(img.shape)), np.max(np.array(img.shape)[np.array(img.shape) < np.max(np.array(img.shape))]),
                3)
                
                if image_size is None:
                    print(" >>>>> No scaling will be performed")
                    #image_size = cropsize[1]#img.shape[0]
                #if scale_size is None:
                #    scaleSize = img.shape[0], img.shape[1]#image_size, image_size  # 224*scale,224*scale
                #    imSize = None#image_size  # image_size*scale #224
                #else:
                #    scaleSize = image_size, image_size  # scale_size,scale_size
                #    imSize = image_size
                # randPos = rnd.choice([0, 50, 100, 200, 300])
                # img = data[k][:, randPos:(randPos+imSize), randPos:(randPos+imSize)] #:32, :32] #size currently (927,1276,3)

                if original_dir:
                    if not os.path.exists(original_dir):
                        os.mkdirs(original_dir)
                    save_original(data[k][:, :, :], save_dir=original_dir, name=str(k))

                ##quality_cropper = quality_random_crop(data[k][:,:, :],imSize)
                ##img = quality_cropper.random_crop_select()



                aspect_ratio = min(img.shape[1] / float(img.shape[0]), img.shape[0] / float(img.shape[1]))
                #acc_scaleSize = (int(scaleSize[0] * aspect_ratio), scaleSize[1]) if img.shape[0] < img.shape[1] else (
                #scaleSize[0], int(aspect_ratio * scaleSize[1]))
                #W = int(imSize * img.shape[1] / float(img.shape[0]))
                #H = int(imSize * img.shape[0] / float(img.shape[1]))

                # print(" True size after scaling to preserve aspect ratio: ", scaleSize)
                # print("larger crop size: ", W," ",H)

                multi_crop_ = 1
                if crop_count < crop_per_case:
                    multi_crop_ = multi_crop + 1
                    crop_count += 1
                for tile in range(multi_crop_):
                    if multi_crop_ != 1:
                        total_crops += 1
                    full_w = img_og.shape[0]
                    full_h = img_og.shape[1]
                    if multi_crop_ > 4:
                        continue
                    start_w = [full_w//10, full_w//5, full_w//3, full_w//2][tile] if multi_crop_ <= 4 else [100, 200, 300, 400, 500, 600][tile]
                    start_h = [full_h//10, full_h//5, full_h//3, full_h//2][tile] if multi_crop_ <= 4 else [400, 400, 400, 400, 400, 400][tile]

                    copy_func = copy.deepcopy if multi_crop_ != 1 else lambda x: x
                    img_crop = copy.deepcopy(img_og)

                    if image_size is not None:
                        if (start_w + 2 * image_size < full_w) and (start_h + 2 * image_size < full_h):
                            img_crop = img_crop[start_w:(start_w + 2 * image_size), start_h:(start_h + 2 * image_size), :]
                        else:
                            img_crop = img_crop[0:image_size, 0:image_size, :]

                    if image_size is not None and (img_crop.shape[0] < image_size or img_crop.shape[1] < image_size):
                        print(" >>>>>>>>>>>>>>>> WARNING: excluding sample, too small")
                        continue
                    # write crop in IDX file format in
                    # Z space filling order for parameter based
                    # resolutuion on read
                    if False and write_crop and mode == 'w':
                        #im_crop = np.transpose(img_og, [2, 0, 1])
                        s=""
                        idx_fname = fname.split('/')[-1].split('.')[0]+'_'+str(k)+'~'+str(label[k])+'.idx'
                        print(os.path.join(idx_filepath,idx_fname))
                        visusWriter.convert_image(image=img_crop, idx_filename=os.path.join(idx_filepath,idx_fname))

                        #im_crop = np.transpose(img_og, [1, 2, 0])
                    # img_stack = np.expand_dims( img_crop, axis=0)
                    if image_size is not None:

                        if img_crop.shape[0] != image_size:
                            img_crop = Image.fromarray(img_crop, 'RGB')
                            # img_crop.show()
                            # import sys
                            # sys.exit(0)
                            img_crop = img_crop.resize((image_size, image_size), Image.ANTIALIAS)
                            img_crop = np.array(img_crop)
                            img_crop = img_crop.reshape((image_size, image_size, 3))  # .transpose(0,3,1,2)

                    # to place as nd array for tensorflow and augmentors
                    if label[k] == 0:
                        # label[k] = 0.1
                        class_0 += 1
                    elif label[k] == 1:
                        # label[k] = 0.9
                        class_1 += 1

                    if label[k] != 0 and label[k] != 1:
                        for i in range(20):
                            print(" >>>>>>>>>>>>>> LABELING INCORRECT> VALUE: ", label[k])

                    if normalize:
                        img_crop = normalize_staining().apply_image(img_crop)

                    ret.append([img_crop.astype("uint8"), label[k]])
                    # img = copy.deepcopy(img_og)

    print(">>>> Total crops observed: ", total_crops)
    return (ret, class_0, class_1, unique_samples)



def get_filenames(dir, train_or_test = '', unknown_dir = None, idx = False):
    filenames = []

    ## Need to write 'read idx filenames!

    if train_or_test == '':
        print("loading passed directy as training dataflow")
        path, dirs, files_train = next(os.walk(os.path.join(dir)))
        file_count = len(files_train)
        filenames = [os.path.join(dir, batch) for batch in files_train]  #
        print(">>>>>>>>>> Using ", str(file_count), " batched files.")

    if train_or_test == 'train':
        print(">>>>>>>>>>>>>>>>  ", os.path.join(dir, 'train','train_idx'))
        path, dirs, files_train = next(os.walk(os.path.join(dir, 'train'))) if not idx else next(os.walk(os.path.join(dir, 'train','train_idx')))
        file_count = len(files_train)
        filenames = [os.path.join(dir, 'train', batch) for batch in files_train] if not idx else [os.path.join(dir, 'train', 'train_idx',batch) for batch in files_train]
        print( ">>>>>>>>>> Using ", str(file_count), " batched files.")

    if train_or_test == 'val':
        path_val, dirs_val, files_val = next(os.walk(os.path.join(dir, 'test'))) if not idx else next(os.walk(os.path.join(dir, 'test','test_idx')))
        file_count_val = len(files_val)
        filenames = [os.path.join(dir, 'test',batch) for batch in files_val] if not idx else [os.path.join(dir, 'test','test_idx',batch) for batch in files_val]#
        print(">>>>>>>>>> Using ", str(file_count_val), " validation batch files.")
    else:
        if unknown_dir:
            path_unknown, dirs_unknown, files_unknown = next(os.walk(os.path.join(dir, 'Unknown',unknown_dir)))
            file_count_unknown = len(files_unknown)
            filenames = [os.path.join(dir, 'Unknown',unknown_dir,f) for f in files_unknown]
    print(" >>>>>>>>>>>>>>>>>>>   total files: ", len(filenames))
    print(filenames)
    return filenames

class lymphomaBase( RNGDataFlow ):
    # Data flow for lymphoma images.
    # yields [ Image, Label ]
    # image: 900x900x3 in range [0,255]
    # label: int either 0 or 1
    def __init__(self, train_or_test, image_size = None, scale_size = None
                 , scale = 2, multi_crop=None, crop_per_case = None, normalize = 0
                 , shuffle=None, dir=None, lymphoma_num_classes=2,unknown_dir = None
                 , original_dir=None, write_crop=True, idx_filepath=None, mode=None
                 , idx=False, resolution=None, memory_profile = None):

        assert train_or_test in ['train', 'test', 'val', '']
        assert lymphoma_num_classes == 2 or lymphoma_num_classes == 10
        self.lymphoma_num_classes = lymphoma_num_classes

        self.shuffle = shuffle

        self.normalize = bool(normalize)
        
        if multi_crop is None:
            multi_crop = 0
        self.multi_crop = multi_crop
        
        self.crop_per_case = crop_per_case
        
        self.train_or_test = train_or_test
        
        if self.train_or_test == 'test':
            self.shuffle = False
            shuffle = False
        
        if dir is None:
            dir = '../data' #and changes in behavor for more classes here
        fnames = get_filenames(dir, train_or_test, unknown_dir=unknown_dir, idx=idx)
        
        self.fs = fnames

        for f in self.fs:
            if not os.path.isfile(f):
                raise ValueError('Failed to find file: ' + f)

        self.train_or_test = train_or_test

        self.scale = scale

        self.image_size = image_size
        self.scale_size = scale_size
        
        print(">> reading in files from: ", original_dir)
        if idx is None:
            data = read_lymphoma(self.fs, train_or_test = self.train_or_test
                                 , image_size = self.image_size, scale_size = self.scale_size
                                 , scale = self.scale, multi_crop=self.multi_crop, crop_per_case = self.crop_per_case
                                 , normalize = self.normalize, original_dir=original_dir) #different classes changes here ect..
        else:
            print(" >>>> writing idx files to: ", idx_filepath)
            data = read_write_idx_lymphoma(self.fs, train_or_test = self.train_or_test
                                 , image_size = self.image_size, scale_size = self.scale_size
                                 , scale = self.scale, multi_crop=self.multi_crop, crop_per_case = self.crop_per_case
                                 , normalize = self.normalize, original_dir=original_dir
                                 ,write_crop=write_crop, idx_filepath=idx_filepath, mode=mode, resolution=resolution) #different classes changes here ect..
        self.data = data[0]
        self.class_0 = data[1]
        self.class_1 = data[2]
        self.unique_samples = data[3]
        
        print("")
        print(">>>> ", train_or_test," set >>>")
        print(">>> Total class 0 samples: ", self.class_0)
        print(">>> Total class 1 samples: ", self.class_1)
        print("")
        self.dir = dir
        self.shuffle = shuffle
        #self.memory_log = self.memory_profile
        
    def size(self):
        return len(self.data)
    
    # Required get_data for DataFlow
    """memory_profile_log = open(memory_log, 'w+')
    @profile(stream=memory_profile_log)"""
    def get_data(self):
        image_data = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(image_data)
        for i in image_data:
            yield self.data[i]

    def __len__(self):
        return len(self.data)

    """memory_profile_log = open(memory_log, 'w+')
    @profile(stream=memory_profile_log)"""
    def __iter__(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            # since cifar is quite small, just do it for safety
            yield self.data[k]
    
    def get_per_pixel_mean(self):
        """
        return a mean image of all (train and test) images of size 927x1276
        """
        fnames = get_filenames(self.dir, self.train_or_test)
        all_imgs = [x[0] for x in read_lymphoma(self.fs)]
        arr = np.array(all_imgs, dtype='float32')
        mean = np.mean(arr, axis=0)
        return mean
    def get_per_channel_mean(self):
        """
        return three values as mean of each channel
        """
        mean = self.get_per_pixel_mean()
        return np.mean(mean, axis=(0, 1))

    def reset_state(self):
        """ Reset the RNG """
        self.rng = get_rng(self)
        super(lymphomaBase, self).reset_state()
        
class lymphoma2(lymphomaBase):
    """
    Produces [image, label] in Cifar10 dataset,
    image is 900x900x3 in the range [0,255].
    label is an int.
    """
    def __init__(self, train_or_test, image_size = None, scale_size = None, scale = None
                 , multi_crop= None, crop_per_case = None, normalize = None, shuffle= None
                 , dir=None, unknown_dir=None,original_dir=None, idx=None):

        """
        Args:
            train_or_test (str): either 'train' or 'test'.
            shuffle (bool): shuffle the dataset.
        """
        if shuffle == None and train_or_test == 'train':
            shuffle = True
        elif shuffle == None and train_or_test == 'test':
            shuffle = False
        self.shuffle = shuffle

        if normalize is None:
            normalize = 1
        self.normalize = bool(normalize)
        
        if multi_crop == None:
            multi_crop = 0
        self.multi_crop = multi_crop
        
        self.crop_per_case = crop_per_case

        self.scale = scale
        self.image_size = image_size
        self.scale_size = scale_size
        
        super(lymphoma2, self).__init__(train_or_test, image_size = self.image_size, scale_size = self.scale_size
                                        , scale=self.scale, multi_crop=self.multi_crop, crop_per_case = self.crop_per_case
                                        , normalize = self.normalize, shuffle = self.shuffle, dir=dir, lymphoma_num_classes = 2
                                        ,unknown_dir = unknown_dir, original_dir = original_dir
                                        , idx=idx,idx_filepath=None, mode=None)

# data converter for dataflow into IDX format
# written to disk in Z space filling order
# for parameter based resolution on read
class lymphoma2ZIDX(lymphomaBase):
    """
    Produces [image, label] in Cifar10 dataset,
    image is 900x900x3 in the range [0,255].
    label is an int.
    """

    def __init__(self, train_or_test, image_size=None, scale_size=None, scale=None, multi_crop=None, crop_per_case=None,
                 normalize=None, shuffle=None, dir=None, unknown_dir=None, original_dir=None
                 ,idx_filepath=None, mode=None, idx=True, resolution=None, memory_profile=None):

        """
        Args:
            train_or_test (str): either 'train' or 'test'.
            shuffle (bool): shuffle the dataset.
        """
        if mode is None:
            print("no read or write mode provided. Assuming read ('r')")
            mode = 'r'
        if memory_profile is not None:
            memory_log = memory_profile

        if shuffle == None and train_or_test == 'train':
            shuffle = True
        elif shuffle == None and train_or_test == 'test':
            shuffle = False
        self.shuffle = shuffle

        if normalize is None:
            normalize = 0
        self.normalize = bool(normalize)

        if multi_crop == None:
            multi_crop = 0
        self.multi_crop = multi_crop

        self.crop_per_case = crop_per_case

        self.scale = scale
        self.image_size = image_size
        self.scale_size = scale_size

        super(lymphoma2ZIDX, self).__init__(train_or_test, image_size=self.image_size, scale_size=self.scale_size
                                        , scale=self.scale, multi_crop=self.multi_crop, crop_per_case=self.crop_per_case
                                        , normalize=self.normalize, shuffle=self.shuffle, dir=dir,
                                        lymphoma_num_classes=2, unknown_dir=unknown_dir, original_dir=original_dir
                                            ,idx_filepath=idx_filepath, mode=mode, idx=True, resolution=resolution, memory_profile = memory_profile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Takes directory of training images or testing images')
    parser.add_argument('-traindir', type=str, nargs = 1, help ='path to training data', required = False)
    parser.add_argument('-testdir', type=str, nargs = 1, help ='path to testing data ', required = False)
    parser.add_argument('-tot', type=str, nargs = 1, help ="Either 'train' or 'test' ", required = False)
    args = parser.parse_args()
    
    if args.traindir[0]:
        dir = args.traindir[0]
    if args.testdir[0]:
        dir = args.testdir[0]
    #print(dir)

    #ds = lymphoma2('train', dir = dir)
    #mean = ds.get_per_channel_mean()
    #print(mean)

    #import cv2
    #ds.reset_state()
    #for i, dp in enumerate(ds.get_data()):
    #    if i == 100:
    #        break
    #    img = dp[0]
    #    cv2.imwrite("{:04d}.jpg".format(i), img)
    # get pixel mean here?
    #print('here')
