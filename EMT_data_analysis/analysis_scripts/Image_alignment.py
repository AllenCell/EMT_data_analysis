######---------importing libraries--------#######
import numpy as np
import pickle as pkl

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