"""
Authors: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import numpy as np
import torch

from utils.config import create_config
from utils.common_config import get_model, get_train_dataset, \
                                get_val_dataset, \
                                get_val_dataloader, \
                                get_val_transformations \
                                
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='Eval_nn')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    
    # Model
    model = get_model(p)
    model = model.cuda()
   
    # CUDNN
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    val_dataset = get_val_dataset(p, val_transforms) 
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {} val samples'.format(len(val_dataset)))
    
    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Checkpoint
    print(p['pretext_checkpoint'])
    assert os.path.exists(p['pretext_checkpoint'])
    print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
    checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint)
    model.cuda()

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    number_of_candidate_images = list(range(50,1001,50))
    for no_of_cand_images in number_of_candidate_images:
        print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
        fill_memory_bank(val_dataloader, model, memory_bank_val)
        topk = no_of_cand_images
        print('Mine the nearest neighbors (Top-%d)' %(topk)) 
        indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
        np.save("repository_eccv\\cifar-20\\pretext\\top"+str(no_of_cand_images)+"-val-neighbors.npy", indices)   

 
if __name__ == '__main__':
    main()
