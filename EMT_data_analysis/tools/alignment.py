import numpy as np
from typing import List, Union
from skimage.transform import SimilarityTransform, warp

import warnings
warnings.filterwarnings("ignore")

def parse_rotation_matrix_from_string(matrix_string: str):
    # Hacky convertion of string to matrix. Maybe there is a better solution (regex?)
    matrix_string = matrix_string.replace("  "," ").replace("[ ","[")
    matrix_string = matrix_string.replace("\n ", ",").replace(" ", ",")
    matrix = np.asarray(eval(matrix_string))
    return matrix

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

def get_alignment_matrix(alignment_matrix: np.ndarray):
    '''
    This function returns the alignment matrix for a given barcode.
    
    Parameters
    ----------
    alignment_matrix: np.ndarray
        Alignment parameters, as a 3x3 matrix, as provided by the "Camera Alignment Matrix" column. 
    '''
    return SimilarityTransform(matrix=alignment_matrix)