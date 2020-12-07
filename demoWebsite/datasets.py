import os
import cv2
import pickle
import numpy as np
import torch
        
class Anime:
    """ Dataset that loads images and image tags from given folders.

    Attributes:
        root_dir: folder containing training images
        tags_file: a dictionary object that contains class tags of images.
        transform: torch.Transform() object to perform image transformations.
        img_files: a list of image file names in root_dir
        dataset_len: number of training images.
    """

    def __init__(self, training_data):        
        self.training_data = training_data
    
    def length(self):
        return len(self.training_data)
    
    def get_item(self, idx):
        return self.training_data[idx]

class Shuffler:
    """ Class that supports andom sampling of training data.

    Attributes:
        dataset: an Anime dataset object.
        batch_size: size of each random sample.
        dataset_len: size of dataset.
    
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_len = self.dataset.length()
    
    def get_batch(self):
        """ Returns a batch of randomly sampled images and its tags. 

        Args:
            None.

        Returns:
            Tuple of tensors: img_batch, hair_tags, eye_tags
            img_batch: tensor of shape N * 3 * 64 * 64
            hair_tags: tensor of shape N * hair_classes
            eye_tags: tensor of shape N * eye_classes
        """

        indices = np.random.choice(self.dataset_len, self.batch_size)  # Sample non-repeated indices
        img_batch, hair_tags, eye_tags, face_tags, glasses_tags = [], [], [], [], []
        for i in indices:
            img, hair_tag, eye_tag, face_tag, glasses_tag = self.dataset.get_item(i)
            img_batch.append(img.unsqueeze(0))
            hair_tags.append(hair_tag.unsqueeze(0))
            eye_tags.append(eye_tag.unsqueeze(0))
            face_tags.append(face_tag.unsqueeze(0))
            glasses_tags.append(glasses_tag.unsqueeze(0))
            
        img_batch = torch.cat(img_batch, 0)
        hair_tags = torch.cat(hair_tags, 0)
        eye_tags = torch.cat(eye_tags, 0)
        face_tags = torch.cat(face_tags, 0)
        glasses_tags = torch.cat(glasses_tags, 0)
        return img_batch, hair_tags, eye_tags, face_tags, glasses_tags
    
