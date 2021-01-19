#import LyCNN.datapack.lymphomaDataPack as datapack
from ..lymphomaDataPack import lymphoma2ZIDX
from ..medical_aug import *
import tensorflow as tf
from tensorpack import *
import numpy as np
import cv2
from PIL import Image

class Convert_Dataset_ZIDX:
    def __init__(self, read_path=None, write_path=None):

        self.dataset = lymphoma2ZIDX(train_or_test='', multi_crop = 2, shuffle = False
                                     , dir=read_path,idx_filepath=write_path
                                     , mode='w')
        #ds = datapack.lymphoma2('train', multi_crop = False, shuffle = False, dir='../data/playground')

        self.batch_size = self.dataset.size()
        print("total im: ", self.batch_size)


        #ds_unAug = BatchData(self.dataset, 1, remainder=True)
        show_idx = False
        if show_idx:
            c = 0
            im = []
            im_aug = []
            label = 2
            for dp_ua in self.dataset:#_unAug:
                print("Data shape OG: ", len(dp_ua), ' elm 0 size ', dp_ua[0].shape)
                if True:
                    im = dp_ua#[0][0,:,:,:]
                    #image_show = Image.fromarray(im[0].astype("uint8"), mode="RGB")
                    #image_show.show()
                    im = im[0]/255.
                    #tf.image.convert_image_dtype(im[0], dtype=tf.float32)
                    #im = im.astype("uint8")
                    #max_elm = max(im)
                    #print(max_elm)
                    image_show = Image.fromarray(im.astype("uint8"), mode="RGB")
                    image_show.show()
                    self.im = dp_ua[0]
                c+=1

    def View_Augmentation(self):
        augmentors = [
            #datapack.HematoEAug((0.8, 0.9, 8)),#np.random.randint(2**32-1))),
            #datapack.HematoEAug((0.1, 0.2, np.random.randint(2**32-1))),
            NormStainAug(False),
            #imgaug.CenterPaste((224,224)),
        ]

        augmentor = imgaug.AugmentorList(augmentors)
        ds = MultiThreadMapData(self.dataset,
                                nr_thread=2,
                                map_func=lambda dp: [augmentor.augment(dp[0]),
                                                            dp[1]],
                                buffer_size=100)

        #ds = AugmentImageComponent(ds, augmentors)
        ds = PrefetchDataZMQ(ds, nr_proc=1)
        ds = BatchData(ds, 1, remainder=True)


        print(">>>>>>>>>>>> size after adding augmentors:  ", ds.size())
        ds.reset_state()

        c = 0
        im_aug = []
        for dp in ds:
            print("Data shape: ", len(dp), ' elm 0 size ', dp[0].shape)
            if c == 0:
                im_aug = dp[0][0,:,:,:]
                label = dp[1]
            c+=1

        #im = im.astype('uint8')
        #im_aug = im_aug.astype('uint8')

        #im_aug_he = augmentors[0]._augment(im)
        im_aug_he_2 = augmentors[0].augment(self.im)
        im_aug_he_3 = augmentors[0].augment(im_aug_he_2)
        im_aug_he = hematoxylin_eosin_aug(1.3, 1.4,8).apply_image(self.im)
        im_aug_norm = normalize_staining().apply_image(im_aug_he)

        im_aug_he_norm = normalize_staining().apply_image(im_aug_he)


        print(" Class of Image: ", label)

        img_aug = Image.fromarray( im_aug )
        img_aug.show( title="H&E Augmentation")

        print("H&E")

        aug_he = Image.fromarray(im_aug_he )
        aug_he.show(title="H&E Augmentation 2")
        #aug_he_2 = Image.fromarray(im_aug_he_2)
        #aug_he_2.show()
        print("H&E 2")

        #aug_norm = Image.fromarray(im_aug_norm )
        #aug_norm.show( title="Normalization Augmentation")

        print("Norm")

        #aug_he_norm = Image.fromarray( im_aug_he_norm )
        #aug_he_norm.show( title="Normalization of H&E Augmentation")

        print("Norm(H&E)")

        img_og = Image.fromarray( self.im )
        img_og.show( title="Original Image")

        print("OG")

