######---------importing libraries--------#######
import numpy as np
import pickle as pkl
from typing import Tuple, List

import warnings
warnings.filterwarnings("ignore")

from skimage.transform import SimilarityTransform, warp


# %% [markdown]
#######----apply alignment matrix to 2D image----####
def align_image(
        img: np.ndarray, 
        transform: SimilarityTransform   
    ):
    '''
    This function aligns an image according to the camera alignment matrix for its barcode.
    
    Parameters
    ----------
    img: np.ndarray
        Image to be aligned. Image assumed to be YX.
    transform: SimilarityTransform
        Transformation matrix to align the image.
    '''
    return warp(img, transform, order=0, preserve_range=True)

def get_alignment_matrix(barcode, alignment_folder='/allen/aics/assay-dev/users/Filip/Data/EMT-alignment-matrices/alignment_info/'):
    '''
    This function returns the alignment matrix for a given barcode.
    
    Parameters
    ----------
    barcode: str
        Barcode of the image.
    alignment_folder: str
        Folder path where alignment matrices are stored.
    '''
    matrix = pkl.load(open(f'{alignment_folder}/{barcode}_alignmentmatrix.pkl', 'rb'))
    return SimilarityTransform(matrix=matrix)

def camera_correction(img: np.ndarray, offset: Tuple[int,int] = (8, 14), top_left_channels: List[int] = [0,3]) -> np.ndarray:
    '''
    This function corrects a camera fov missalignment that occured in wells imaging N-Cad and E-Cad.
    The the missalignment is corrected by cropping the top left and bottom right corners of channels
    1/4 and 2/3/5, respectively so that only the overlapping fov remains.
    
    Parameters
    ----------
    img: np.ndarray
        The 4D/3D (C,Z,Y,X)/(Z,Y,X) image that needs camera offset to be corrected. If the image is
            4D it will apply to each channel as appropriately. For 3D (assumed brightfield mask) it
            will crop only the top-left corner
    offset: Tuple(int,int)
        Offset of the camera fov's. Default: (8,14)
    top_left_channels: List[int]
        Which channels need to be cropped from the top left. All other channels will be cropped from
        the bottom right. Note: A 3D image is always treated as being channel 0. Default: [0,3]
    ''' 
    assert img.ndim == 4 or img.ndim == 3, f"Invalid image shape {img.ndim}D. Must be 3D or 4D."

    def crop(img_ch: np.ndarray, top_left=True):
        if top_left:
            return img_ch[:, offset[0]+1:, offset[1]+1:]
        else:
            return img_ch[:, :-offset[0], :-offset[1]]
    
    if img.ndim == 3:
        img = np.expand_dims(img, 0)

    # Indeces of channels in camera 1
    return np.stack([crop(img[ch], ch in top_left_channels) for ch in range(img.shape[0])], axis=0).squeeze()

