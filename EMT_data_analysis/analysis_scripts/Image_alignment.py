######---------importing libraries--------#######
import numpy as np
from typing import List, Union

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

def get_alignment_matrix(alignment_matrix:Union[np.ndarray, str]):
    '''
    This function returns the alignment matrix for a given barcode.
    
    Parameters
    ----------
    alignment_matrix: Union[np.ndarray, str]
        Alignment parameters, as a 3x3 matrix, as provided by the "Camera Alignment Matrix" column. 
        Can either be the original string from the manifest or already converted to an ndarray
    '''
    if isinstance(alignment_matrix, str):
        alignment_matrix = np.ndarray(eval(alignment_matrix))
    return SimilarityTransform(matrix=alignment_matrix)