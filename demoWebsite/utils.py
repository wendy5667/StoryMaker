import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image
from demoWebsite.wordbase import face_color, eye_color, hair_color, face_type, nega
import random

# Number to text mapping of class labels
hair_mapping =  ['blonde', 'orange', 'brown', 'black', 'blue', 'white']

eye_mapping = ['brown', 'blue', 'green', 'black']

face_mapping = ['African', 'Asian', 'Caucasian']

glasses_mapping = ['with_glasses', 'without_glasses']

def encodeJSON(infos):
    hair = [0, 0, 0, 0, 0, 0]
    eye = [0, 0, 0, 0]
    face = [0, 0, 0]
    glasses = [0, 0]
    
    for info in infos:
        if info["topScoringIntent"]["intent"] == "VR.None" or info["topScoringIntent"]["score"] < 0.3:
            continue
        if info["topScoringIntent"]["intent"] == "VR.FaceColor":
            neg_flag = False
            set_color = -1
            for ent in info["entities"]:
                if ent["entity"] in face_color:
                    set_color = face_color[ent["entity"]]
                elif ent["entity"] in face_type:
                    set_color = face_type[ent["entity"]]
                elif ent["entity"] in nega:
                    neg_flag = True
                else:
                    continue
            if set_color == -1:
                continue
            else: 
                if neg_flag:
                    face[set_color] = -1
                else: 
                    face[set_color] = 1

        elif info["topScoringIntent"]["intent"] == "VR.EyeColor":
            neg_flag = False
            set_color = -1
            for ent in info["entities"]:
                if ent["entity"] in eye_color:
                    set_color = eye_color[ent["entity"]]
                elif ent["entity"] in nega:
                    neg_flag = True
                else:
                    continue
            if set_color == -1:
                continue
            else: 
                if neg_flag:
                    eye[set_color] = -1
                else: 
                    eye[set_color] = 1
        elif info["topScoringIntent"]["intent"] == "VR.HairColor":
            neg_flag = False
            set_color = -1
            for ent in info["entities"]:
                if ent["entity"] in hair_color:
                    set_color = hair_color[ent["entity"]]
                elif ent["entity"] in nega:
                    neg_flag = True
                else:
                    continue
            if set_color == -1:
                continue
            else: 
                if neg_flag:
                    hair[set_color] = -1
                else: 
                    hair[set_color] = 1
        elif info["topScoringIntent"]["intent"] == "VR.Glasses":
            neg_flag = False
            for ent in info["entities"]:
                if ent["entity"] in nega:
                    neg_flag = True
                else:
                    continue
            
            if neg_flag:
                glasses[1] = 1
            else: 
                glasses[0] = 1
        else:
            sys.exit("Function not found")

    features = []
    glasses_ = glasses
    for n in range(5):
        if 1 in face:
            idx = face.index(1)
        else: 
            idx = random.choice([i for i in range(len(face)) if face[i]!=-1])

        face_ = [1 if i == idx else 0 for i in range(len(face))]
        if 1 in eye:
            idx = eye.index(1)
        else: 
            idx = random.choice([i for i in range(len(eye)) if eye[i]!=-1 ])
        eye_ = [1 if i == idx else 0 for i in range(len(eye))]
        if 1 in hair:
            idx = hair.index(1)
        else: 
            idx = random.choice([i for i in range(len(hair)) if hair[i]!=-1 ])
        hair_ = [1 if i == idx else 0 for i in range(len(hair))]
        
        if 1 not in glasses_:
            idx = random.choice([i for i in range(len(glasses))])
            glasses_ = [1 if i == idx else 0 for i in range(len(glasses))]
        else:
            glasses_ = glasses

        print("111111111111111111111111")
        print(glasses_)
        feature = hair_+eye_+face_+glasses_
        print(feature)
        features.append(feature)
    return features


def denorm(img):
    """ Denormalize input image tensor. (From [0,1] -> [-1,1]) 
    
    Args:
        img: input image tensor.
    """
	
    output = img / 2 + 0.5
    return output.clamp(0, 1)

def save_model(model, optimizer, step, log, file_path):
    """ Save model checkpoints. """

    state = {'model' : model.state_dict(),
             'optim' : optimizer.state_dict(),
             'step' : step,
             'log' : log}
    torch.save(state, file_path)
    return

def load_model(model, optimizer, file_path):
    """ Load previous checkpoints. """

    prev_state = torch.load(file_path)
    
    model.load_state_dict(prev_state['model'])
    optimizer.load_state_dict(prev_state['optim'])
    start_step = prev_state['step']
    log = prev_state['log']
    
    return model, optimizer, start_step, log
    
def show_process(total_steps, step_i, g_log, d_log, classifier_log):
    """ Show relevant losses during training. """

    print('Step {}/{}: G_loss [{:8f}], D_loss [{:8f}], Classifier loss [{:8f}]'.format(
            step_i, total_steps, g_log[-1], d_log[-1], classifier_log[-1]))
    return

def plot_loss(g_log, d_log, file_path):
    """ Plot generator and discriminator losses. """

    steps = list(range(len(g_log)))
    plt.semilogy(steps, g_log)
    plt.semilogy(steps, d_log)
    plt.legend(['Generator Loss', 'Discriminator Loss'])
    plt.title("Loss ({} steps)".format(len(steps)))
    plt.savefig(file_path)
    plt.close()
    return

def plot_classifier_loss(log, file_path):
    """ Plot auxiliary classifier loss. """
    
    steps = list(range(len(log)))
    plt.semilogy(steps, log)
    plt.legend(['Classifier Loss'])
    plt.title("Classifier Loss ({} steps)".format(len(steps)))
    plt.savefig(file_path)
    plt.close()
    return

def get_random_label(batch_size, 
                     hair_classes, 
                     eye_classes, 
                     face_classes, 
                     glasses_classes):
    """ Sample a batch of random class labels given the class priors.
    
    Args:
        batch_size: number of labels to sample.
        hair_classes: number of hair colors. 
        hair_prior: a list of floating points values indicating the distribution
					      of the hair color in the training data.
        eye_classes: (similar as above).
        eye_prior: (similar as above).
    
    Returns:
        A tensor of size N * (hair_classes + eye_classes). 
    """
    
    hair_code = torch.zeros(batch_size, hair_classes)  # One hot encoding for hair class
    eye_code = torch.zeros(batch_size, eye_classes)  # One hot encoding for eye class
    face_code = torch.zeros(batch_size, face_classes)
    glasses_code = torch.zeros(batch_size, glasses_classes)

    hair_type = np.random.choice(hair_classes, batch_size)  # Sample hair class from hair class prior
    eye_type = np.random.choice(eye_classes, batch_size)  # Sample eye class from eye class prior
    face_type = np.random.choice(face_classes, batch_size)
    glasses_type = np.random.choice(glasses_classes, batch_size)
    
    for i in range(batch_size):
        hair_code[i][hair_type[i]] = 1
        eye_code[i][eye_type[i]] = 1
        face_code[i][face_type[i]] = 1
        glasses_code[i][glasses_type[i]] = 1

    return torch.cat((hair_code, eye_code, face_code, glasses_code), dim = 1) 

def generation_by_attributes(model, device, latent_dim, batch_size, hair_classes, eye_classes, face_classes, glasses_classes,
    sample_dir, step = None, fix_hair = None, fix_eye = None):
    """ Generate image samples with fixed attributes.
    
    Args:
        model: model to generate images.
        device: device to run model on.
        step: current training step. 
        latent_dim: dimension of the noise vector.
        fix_hair: Choose particular hair class. 
                  If None, then hair class is chosen randomly.
        hair_classes: number of hair colors.
        fix_eye: Choose particular eye class. 
                 If None, then eye class is chosen randomly.
        eye_classes: number of eye colors.
        sample_dir: folder to save images.
    
    Returns:
        None
    """
    
    hair_tag = torch.zeros(batch_size, hair_classes).to(device)
    eye_tag = torch.zeros(batch_size, eye_classes).to(device)
    face_tag = torch.zeros(batch_size, face_classes).to(device)
    glasses_tag = torch.zeros(batch_size, glasses_classes).to(device)
    
    hair_class = np.random.randint(hair_classes)
    eye_class = np.random.randint(eye_classes)
    face_class = np.random.randint(face_classes)
    glasses_class = np.random.randint(glasses_classes)

    for i in range(batch_size):
        hair_tag[i][hair_class] = 1
        eye_tag[i][eye_class] = 1
        face_tag[i][face_class] = 1
        glasses_tag[i][glasses_class] = 1
        
    tag = torch.cat((hair_tag, eye_tag, face_tag, glasses_tag), 1)
    z = torch.randn(batch_size, latent_dim).to(device)

    output = model(z, tag)
    if step is not None:
        file_path = '{} hair {} eyes {} face {}, step {}.png'.format(hair_mapping[hair_class], 
                                                          eye_mapping[eye_class], 
                                                          face_mapping[face_class], 
                                                          glasses_mapping[glasses_class], step)
    else:
        file_path = '{} hair {} eyes {} face {}.png'.format(hair_mapping[hair_class], 
                                                            eye_mapping[eye_class], 
                                                            face_mapping[face_class], 
                                                            glasses_mapping[glasses_class])
    save_image(denorm(output), os.path.join(sample_dir, file_path))


def hair_grad(model, device, latent_dim, hair_classes, eye_classes, file_path):
    """ Generate image samples with fixed eye class and noise, change hair color.
    
    Args:
        model: model to generate images.
        device: device to run model on.
        latent_dim: dimension of the noise vector.
        hair_classes: number of hair colors.
        eye_classes: number of eye colors.
        sample_dir: folder to save images.
    
    Returns:
        None
    """

    eye = torch.zeros(eye_classes).to(device)
    eye[np.random.randint(eye_classes)] = 1
    eye.unsqueeze_(0)
    
    z = torch.randn(batch_size, latent_dim).to(device)
    img_list = []
    for i in range(hair_classes):
        hair = torch.zeros(hair_classes).to(device)
        hair[i] = 1
        hair.unsqueeze_(0)
        tag = torch.cat((hair, eye), 1)
        img_list.append(model(z, tag))

    output = torch.cat(img_list, 0)
    save_image(utils.denorm(output), file_path, nrow = hair_classes)

def eye_grad(model, device, latent_dim, hair_classes, eye_classes, file_path):
    """ Generate random image samples with fixed hair class and noise, change eye color.
    
    Args:
        model: model to generate images.
        device: device to run model on.
        latent_dim: dimension of the noise vector.
        hair_classes: number of hair colors.
        eye_classes: number of eye colors.
        file_path: output file path.
    
    Returns:
        None
    """

    hair = torch.zeros(hair_classes).to(device)
    hair[np.random.randint(hair_classes)] = 1
    hair.unsqueeze_(0)

    z = torch.randn(batch_size, latent_dim).to(device)
    img_list = []
    for i in range(eye_classes):
        eye = torch.zeros(eye_classes).to(device)
        eye[i] = 1
        eye.unsqueeze_(0)
        tag = torch.cat((hair, eye), 1)
        img_list.append(model(z, tag))
        
    output = torch.cat(img_list, 0)
    save_image(utils.denorm(output), file_path, nrow = eye_classes)

def fixed_noise(model, device, latent_dim, hair_classes, eye_classes, file_path):
    """ Generate random image samples with fixed noise.
    
    Args:
        model: model to generate images.
        device: device to run model on.
        latent_dim: dimension of the noise vector.
        hair_classes: number of hair colors.
        eye_classes: number of eye colors.
        file_path: output file path.
    
    Returns:
        None
    """
    
    z = torch.randn(batch_size, latent_dim).to(device)
    img_list = []
    for i in range(eye_classes):
        for j in range(hair_classes):
            eye = torch.zeros(eye_classes).to(device)
            hair = torch.zeros(hair_classes).to(device)
            eye[i], hair[j] = 1, 1
            eye.unsqueeze_(0)
            hair.unsqueeze_(0)

            tag = torch.cat((hair, eye), 1)
            img_list.append(model(z, tag))
        
    output = torch.cat(img_list, 0)
    save_image(utils.denorm(output), file_path, nrow = eye_classes)