import numpy as np
from scipy.signal import convolve2d
import scipy
def remove_stuck_particle_2(original_img,M,nbr_its):
    original_img = original_img[0,:,:,0]
    #plt.figure()
    #plt.imshow(original_img,aspect='auto')
    for i in range(0,nbr_its):

        conv_img = original_img - convolve2d(original_img,np.ones((M,1))/M,mode='same',boundary='symm',fillvalue=1)
        
        img = original_img * (1-conv_img)
        try:
            img /= np.max(img)
        except:
            return original_img
        
        img[img<0.99] = 0
        img[img>0] = 1

        identifiedStuckTraj = np.sum(img,0)
        img[:,identifiedStuckTraj<2] = 0
        
        binary_img = scipy.ndimage.morphology.binary_dilation(img,structure=np.ones((M,1)))

        idcs = np.sum(binary_img,axis=0)==0
        cut_img = original_img[:,idcs]
        original_img = np.copy(cut_img)
    #plt.figure()
    #plt.imshow(original_img,aspect='auto')
    return np.expand_dims(original_img,(0,-1))