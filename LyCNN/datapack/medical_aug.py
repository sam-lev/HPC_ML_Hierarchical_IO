#!/usr/bin/env python3
"""Extract deep CNN features from a set of images and dump them as Numpy arrays image_file_name.npy"""

from tensorpack.dataflow import imgaug
#try:
#    Transform = imgaug.transform.ImageTransform
#except AttributeError:
#    Transform = imgaug.Transform
import argparse

import cv2
from scipy import ndimage
from os.path import basename, join, exists
from os import makedirs
#from threaded_generator import threaded_generator
from time import time
import sys
import copy as copy_dp

import numpy as np
#np.random.seed(101)

PATCH_SIZES = [224, 224] #[672,672]
SCALES = [0.5]

DEFAULT_INPUT_DIR = "data/train"
DEFAULT_PREPROCESSED_ROOT = "data/preprocessed/train"

PATCHES_PER_IMAGE = 20
AUGMENTATIONS_PER_IMAGE = 50
COLOR_LO = 0.7
COLOR_HI = 1.3
BATCH_SIZE = 16     # decrease if necessary

NUM_CACHED = 160

notification_norm = 0
notification_he = 0
notification_zoom = 0

class NormStainAug(imgaug.ImageAugmentor):
    def __init__(self, param=(True,1.0)):
        self.copy = param[0]
        self.with_prob = param[1]
        super(NormStainAug, self).__init__()
        self._init(locals())
        
    def reset_state(self):
        super(NormStainAug, self).reset_state()
    
    def get_transform(self, _):
        return normalize_staining()
    
    def apply_coords(self, coords):
        return coords

    def _augment_return_params(self, img):
        p_holder = np.array([0])
        p = np.random.RandomState(None).uniform(low=0, high=1, size=1)[0]
        if p <= self.with_prob:
            copy_func = copy_dp.deepcopy if self.copy else lambda x: x
            img_dup = copy_func(img)#.astype("uint8"))
            t = self.get_transform(p_holder).apply_image(img_dup)#[0])
            return t, (self.copy, self.with_prob)
        else:
            return img, (self.copy, self.with_prob)
    
    def _get_augment_params(self, img):
        return (self.copy, self.with_prob)
    
    def _augment(self, img, prms):
        p_holder = np.array([0])
        self.copy = prms[0]
        self.with_prob = prms[1]
        p = np.random.RandomState(None).uniform(low=0, high=1.0, size=1)[0]
        if p <= self.with_prob:
            copy_func = copy_dp.deepcopy if self.copy else lambda x: x
            img_dup = copy_func(img.astype("uint8"))
            t = self.get_transform(p_holder).apply_image(img_dup)
            return t
        else:
            return img
    
    def augment(self, img):
        p_holder = np.array([0])
        p = np.random.RandomState(None).uniform(low=0, high=1, size=1)[0]
        if p <= self.with_prob:
            copy_func = copy_dp.deepcopy if self.copy else lambda x: x
            img_dup = copy_func(img.astype("uint8"))
            t = self.get_transform(p_holder).apply_image(img_dup)
            return t
        else:
            return img

class ZoomAug(imgaug.ImageAugmentor):
    def __init__(self, param = (10, None, True)):
        super(ZoomAug, self).__init__()
        self.zoom = param[0]
        self.seed = param[1]# np.random.randint(2**32-1)
        self.copy = param[2]
        self._init(locals())
        
    def	reset_state(self):
        super(ZoomAug, self).reset_state()
        self.copy = self.copy
        self.seed = None#np.random.randint(2**32-1)
        
    def get_transform(self, _):
        return zoom_transform(self.zoom, self.seed)
    
    def	apply_coords(self, coords):
        return coords
    
    def _get_augment_params(self, img):
        return (self.zoom, self.seed, self.copy)
    
    def	_augment(self, img, param):
        self.zoom = param[0]
        self.seed = param[1]
        self.copy = param[2]
        p_holder = np.array([0])
        copy_func = copy_dp.deepcopy if self.copy else lambda x: x
        img_dup = copy_func(img.astype("uint8"))                             
        t = self.get_transform(p_holder).apply_image(img_dup)
        return t
    
    def	augment(self, img):
        p_holder = np.array([0])
        copy_func = copy_dp.deepcopy if self.copy else lambda x: x
        img_dup = copy_func(img.astype("uint8"))
        t = self.get_transform(p_holder).apply_image(img_dup)
        return t
    
    def _augment_return_params(self, img):
        p_holder = np.array([0])
        copy_func = copy_dp.deepcopy if self.copy else lambda x: x
        img_dup = copy_func(img.astype("uint8"))
        t = self.get_transform(p_holder).apply_image(img_dup)
        return t, (self.zoom, self.seed, self.copy)

class HematoEAug(imgaug.ImageAugmentor):
    def __init__(self, param = (0.5, 1.7, None, True, 1.0)):
        super(HematoEAug, self).__init__()
        self.low = param[0]
        self.high = param[1]
        self.seed = param[2]
        self.copy = param[3]
        self.with_prob = param[4]
        self._init(locals())
        
    def	reset_state(self):
        super(HematoEAug, self).reset_state()
        self.seed = None#np.random.randint(2**32-1)
    
    def get_transform(self, _):
        return hematoxylin_eosin_aug(self.low, self.high, self.seed)
    
    def apply_coords(self, coords):
        return coords
    
    def _get_augment_params(self, img):
        return (self.low, self.high, self.seed, self.copy, self.with_prob)
    
    def	_augment(self, img, param = (0.5, 1.7, None, True, 1.0)):
        self.low = param[0]
        self.high = param[1]
        self.copy = param[3]
        self.seed = param[2]
        self.with_prob = param[4]
        p_holder = np.array([0])
        p = np.random.RandomState(None).uniform(low=0, high=1.0, size=1)[0]
        if p <= self.with_prob:
            copy_func = copy_dp.deepcopy if self.copy else lambda x: x
            img_dup = copy_func(img.astype("uint8"))
            t = self.get_transform(p_holder).apply_image(img_dup)
            return t
        else:
            return img

    def augment(self, img):
        p_holder = np.array([0])
        p = np.random.RandomState(None).uniform(low=0, high=1.0, size=1)[0]
        if p <= self.with_prob:
            copy_func = copy_dp.deepcopy if self.copy else lambda x: x
            img_dup = copy_func(img.astype("uint8"))
            t = self.get_transform(p_holder).apply_image(img_dup)
            return t
        else:
            return img
    
    def _augment_return_params(self, img):
        p_holder = np.array([0])
        copy_func = copy_dp.deepcopy if self.copy else lambda x: x
        img_dup = copy_func(img.astype("uint8"))
        t = self.get_transform(p_holder).apply_image(img_dup)

        p = np.random.RandomState(None).uniform(low=0, high=1.0, size=1)[0]
        if p <= self.with_prob:
            copy_func = copy_dp.deepcopy if self.copy else lambda x: x
            img_dup = copy_func(img.astype("uint8"))
            t = self.get_transform(p_holder).apply_image(img_dup)
            return t, (self.low, self.high, self.seed, self.copy, self.with_prob)
        else:
            return img, (self.low, self.high, self.seed, self.copy, self.with_prob)

class normalize_staining(imgaug.Transform):
    def __init__(self, copy = True):
        super(normalize_staining, self).__init__()
        self.copy = copy
        self._init(locals())
        
    def apply_image(self, img):
        
        """
        Referenced from: https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py
        Example use: view_data_sample.py
        Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
        Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
        
        Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
        """
        
        def implementation2(img):
            Io = 240.
            alpha = 1
            beta=0.15
            
            HERef = np.array([[0.5626, 0.2159],
                              [0.7201, 0.8012],
                              [0.4062, 0.5581]])
            
            maxCRef = np.array([1.9705, 1.0308])
            
            # define height and width of image
            h, w, c = img.shape
            
            # reshape image
            img = img.reshape((-1,3))
            
            # calculate optical density
            OD = -np.log((img.astype("uint8")+1.)/Io)
            
            # remove transparent pixels
            ODhat = OD[~np.any(OD<beta, axis=1)]
            #ODhat = OD[(OD >= beta).all(axis=1)]
            # compute eigenvectors
            eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
            #eigvecs *= -1
            
            #project on the plane spanned by the eigenvectors corresponding to the two 
            # largest eigenvalues
            That = ODhat.dot(eigvecs[:,1:3])
            
            phi = np.arctan2(That[:,1],That[:,0])
            
            minPhi = np.percentile(phi, alpha)
            maxPhi = np.percentile(phi, 100-alpha)
            
            vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
            vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
            
            # a heuristic to make the vector corresponding to hematoxylin first and the 
            # one corresponding to eosin second
            if vMin[0] > vMax[0]:
                HE = np.array((vMin[:,0], vMax[:,0])).T
            else:
                HE = np.array((vMax[:,0], vMin[:,0])).T
            
            # rows correspond to channels (RGB), columns to OD values
            Y = np.reshape(OD, (-1, 3)).T
            
            # determine concentrations of the individual stains
            C = np.linalg.lstsq(HE,Y, rcond=None)[0]
            
            # normalize stain concentrations
            maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
            tmp = np.divide(maxC,maxCRef)
            C2 = np.divide(C,tmp[:, np.newaxis])
            
            # recreate the image using reference mixing matrix
            Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
            Inorm[Inorm>255] = 254
            Inorm = np.reshape(Inorm.T, (h, w, 3)).astype("uint8")  
            
            # unmix hematoxylin and eosin
            H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
            H[H>255] = 254
            H = np.reshape(H.T, (h, w, 3)).astype("uint8")
            
            E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
            E[E>255] = 254
            E = np.reshape(E.T, (h, w, 3)).astype("uint8")
            
            #if saveFile is not None:
            #    Image.fromarray(Inorm).save(saveFile+'.png')
            #    Image.fromarray(H).save(saveFile+'_H.png')
            #    Image.fromarray(E).save(saveFile+'_E.png')
            return Inorm, H, E
        
        Io = 240.
        beta = 0.15
        alpha = 1
        HERef = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        maxCRef = np.array([1.9705, 1.0308])
        
        h, w, c = img.shape
        img = img.reshape(h * w, c)
        OD = -np.log((img.astype("uint8") + 1.) / Io) #img.astype("uint16")
        ODhat = OD[(OD >= beta).all(axis=1)]
        W, V = np.linalg.eig(np.cov(ODhat, rowvar=False))
        
        Vec = -V.T[:2][::-1].T  # desnecessario o sinal negativo
        That = np.dot(ODhat, Vec)
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
        vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax])
        else:
            HE = np.array([vMax, vMin])
        
        HE = HE.T
        Y = OD.reshape(h * w, c).T
        
        C = np.linalg.lstsq(HE, Y, rcond=None)
        maxC = np.percentile(C[0], 99, axis=1)
        
        C = C[0] / maxC[:, None]
        C = C * maxCRef[:, None]
        Inorm = Io * np.exp(-np.dot(HERef, C))
        Inorm = Inorm.T.reshape(h, w, c).clip(0, 255).astype("uint8")
        
        # unmix hematoxylin and eosin                                                                                      
        #H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))         
        #H[H>255] = 254               
        #H = np.reshape(H.T, (h, w, 3)).astype("uint8")                                         
        #E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))         
        #E[E>255] = 254                                                                                     
        #E = np.reshape(E.T, (h, w, 3)).astype("uint8")
        return Inorm#, H, E
    
    def apply_coords(self, coords):
        return coords

class hematoxylin_eosin_aug(imgaug.Transform):
    def __init__(self, low=0.7, high=1.3, seed=None, copy=True):
        super(hematoxylin_eosin_aug, self).__init__()
        self.copy = copy
        self.low = low
        self.high = high
        self.seed = seed
        self._init(locals())
        
    def apply_image(self, img):
        low = self.low
        high = self.high
        seed = self.seed
        #if self.copy:
        #    img_copy = copy.deepcopy(img)
        """
        "Quantification of histochemical staining by color deconvolution"
        Arnout C. Ruifrok, Ph.D. and Dennis A. Johnston, Ph.D.
        http://www.math-info.univ-paris5.fr/~lomn/Data/2017/Color/Quantification_of_histochemical_staining.pdf
        Performs random hematoxylin-eosin augmentation
        # Arguments
        img: Numpy image array.
        low: Low boundary for augmentation multiplier
        high: High boundary for augmentation multiplier
        # Returns
        Augmented Numpy image array.
        """
        D = np.array([[1.88, -0.07, -0.60],
                      [-1.02, 1.13, -0.48],
                      [-0.55, -0.13, 1.57]])
        M = np.array([[0.65, 0.70, 0.29],
                      [0.07, 0.99, 0.11],
                      [0.27, 0.57, 0.78]])
        Io = 240.
        
        h, w, c = img.shape
        OD = -np.log10((img.astype("uint8") + 1.) / Io)#.astype("uint16")
        C = np.dot(D, OD.reshape(h * w, c).T).T
        r = np.ones(3)
        r[:2] = np.random.RandomState(seed).uniform(low=low, high=high, size=2)
        img_aug = np.dot(C, M) * r
        
        img_aug = Io * np.exp(-img_aug * np.log(10)) - 1
        img_aug = img_aug.reshape(h, w, c).clip(0, 255).astype("uint8")
        
        return img_aug
    
    def apply_coords(self, coords):
        return coords

class zoom_transform(imgaug.Transform):
    def __init__(self, zoom, seed = None, copy = True):
        super(zoom_transform, self).__init__()
        self.zoom = zoom
        self.seed = seed
        self.copy = copy
        self._init(locals())
        
    def apply_image(self, img):
        zoom = self.zoom
        seed = self.seed
        #if self.copy:
        #    img_copy = copy.deepcopy(img)
        """Performs a random spatial zoom of a Numpy image array.
        # Arguments
        img: Numpy image array.
        zoom_var: zoom range multiplier for width and height.
        seed: Random seed.
        # Returns
        Zoomed Numpy image array.
        """
        scale = np.random.RandomState(seed).uniform(low=1 / zoom_var, high=zoom_var)
        resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return resized_img
    
    def apply_coords(self, coords):
        return coords
