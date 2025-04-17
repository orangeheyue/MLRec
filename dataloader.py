# Copyright (c) 2025 OrangeAI Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Author: orange
# Time: April 16, 2025
# Task: a dataloader uploader for multimodal reccomendation 

import pickle
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp 

class Dataloader(object):
    '''
        Args:
            data_dir:str : dataset root dir, ./dataset/E-commerce/tiktok
            train_data_path:str , ./dataset/E-commerce/tiktok/trnMat.pkl
            eval_data_path:str ,  ./dataset/E-commerce/tiktok/valMat.pkl
            test_data_path:str ,  ./dataset/E-commerce/tiktok/tstMat.pkl
            multimodal_feature_path_dict: {
                'image': /media/data1/hy/MLRec/dataset/E-commerce/tiktok/image_feat.npy
                'text' : /media/data1/hy/MLRec/dataset/E-commerce/tiktok/text_feat.npy
                'audio' : /media/data1/hy/MLRec/dataset/E-commerce/tiktok/audio_feat.npy
            }
            mode:str , 'train' 'eval' 'test'
        Desc: 
            There are multi-modal features in every dataset.
    '''

    def __init__(self,
                data_dir:str,
                train_data_path:str,
                eval_data_path:str,
                test_data_path:str,
                multimodal_feature_path_dict:dict,
                mode:str
                 ):
        self.data_dir = data_dir
        self.train_data_path = train_data_path
        self.eval_data_path =  eval_data_path
        self.test_data_path = test_data_path
        self.multimodal_feature_path_dict = multimodal_feature_path_dict
        self.mode = mode

        self.train_user_ui_matrix = Dataloader.load_pickle_file(self.train_data_path)
        self.eval_user_ui_matrix  = Dataloader.load_pickle_file(self.eval_data_path)
        self.test_user_ui_matrix  = Dataloader.load_pickle_file(self.test_data_path)

        self.train_user_num, self.train_item_num = self.train_user_ui_matrix.shape[0], self.train_user_ui_matrix[1]


    @staticmethod()
    def load_pickle_file(self, file_path):
        '''
        Load origin user-item interaction matrix from pickle binary file.
        Args:
            file_path (str): Path to the pickle file containing the interaction matrix.
        Returns:
            scipy.sparse.coo_matrix: The loaded matrix in COO format.
        Raises:
            FileNotFoundError: If the file doesn't exist.
            pickle.PickleError: If the file is not a valid pickle file.
            ValueError: If the loaded data cannot be converted to a sparse matrix.
        '''
        try:
            # Try to open and read the file
            with open(file_path, 'rb') as f:
                try:
                    # Load data from pickle file
                    loaded_data = pickle.load(f)
                    # Convert to CSR matrix (non-zero entries as float32)
                    data_csr_matrix = (loaded_data != 0).astype(np.float32)
                    # Convert to COO format
                    data_coo_matrix = sp.coo_matrix(data_csr_matrix)
                    return data_coo_matrix
                    
                except pickle.PickleError as e:
                    raise pickle.PickleError(f"Failed to unpickle the file: {str(e)}")
                except AttributeError as e:
                    raise ValueError(f"Loaded data cannot be converted to sparse matrix: {str(e)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} does not exist")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")
        

    