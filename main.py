from comet_ml import Experiment
from model.baseline_model import BaselineModel
import torch
from utils.CIL_dataset import CILSetTask
from model.temporalShiftModule.ops.transforms import *
import argparse
import yaml, pickle
import torch.nn as nn
import os
import random
import torch.nn.functional as F
random.seed(10)

class TextualContrastiveLoss(nn.Module):
    
    def __init__(self, batch_size, device, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=device)).float())
    
    def forward(self, emb_i, emb_j, labels):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        labels = labels.type(torch.FloatTensor)
        labels = labels.to(device)
        label_t = torch.cat([labels, labels], dim=0)
        ws = torch.eq(label_t.unsqueeze(1), label_t.unsqueeze(0))
        ws = ws*self.negatives_mask

        nominator = torch.sum(ws * torch.exp(similarity_matrix / self.temperature), dim = 1)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

class ContrastiveLoss(nn.Module):
    
    def __init__(self, batch_size, device, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=device)).float())
          
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

def parse_conf(conf, new_dict = {}):
    for k, v in conf.items():
        if type(v) == dict:
            new_dict = parse_conf(v, new_dict)
        else:
            new_dict[k] = v
    return new_dict

def main():
    
    global dict_conf, device, experiment, memory_size, batch_size, is_activityNet
    
    parser = argparse.ArgumentParser(description="vCLIMB Model")
    parser.add_argument("-conf","--conf_path", default = './conf/conf_CILTNT_UCF101.yaml')
    args = parser.parse_args()
    conf_file = open(args.conf_path, 'r')
    print("Conf file dir: ",conf_file)
    dict_conf = yaml.load(conf_file)

    name_comet = dict_conf['comet']['name']
    dict_conf['comet']['name'] = name_comet.format(
        dict_conf['dataset']['name'], 
        dict_conf['feature_encoder']['type'], 
        dict_conf['feature_encoder']['num_segments'], 
        dict_conf['type_task'], 
        dict_conf['memory']['memory_size'], 
        dict_conf['type_loss'])
    
    path_memory = dict_conf['memory']['path_memory']
    dict_conf['memory']['path_memory'] = path_memory.format(
        dict_conf['dataset']['name'], 
        dict_conf['feature_encoder']['type'], 
        dict_conf['feature_encoder']['num_segments'], 
        dict_conf['type_task'], 
        dict_conf['memory']['memory_size'], 
        dict_conf['type_loss'])
    
    path_model = dict_conf['checkpoints']['path_model']
    dict_conf['checkpoints']['path_model'] = path_model.format(
        dict_conf['dataset']['name'], 
        dict_conf['feature_encoder']['type'], 
        dict_conf['feature_encoder']['num_segments'], 
        dict_conf['type_task'], 
        dict_conf['memory']['memory_size'], 
        dict_conf['type_loss'])

    
    api_key = dict_conf['comet']['api_key']
    workspace = dict_conf['comet']['workspace']
    project_name = dict_conf['comet']['project_name']
    experiment = Experiment(api_key=api_key,
                            project_name=project_name, workspace=workspace)
    experiment.log_parameters(parse_conf(dict_conf))
    experiment.set_name(dict_conf['comet']['name'])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = BaselineModel(device, dict_conf, experiment)
    
    path_data = dict_conf['dataset']['path_data']
    with open(path_data, 'rb') as handle:
        data = pickle.load(handle)
        
    num_class = len(data['train'][0].keys())
    
    is_activityNet = dict_conf['dataset']['is_activityNet'] if 'is_activityNet' in dict_conf['dataset'] else False

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    
    dataset_name = dict_conf['dataset']['name']
    
    train_augmentation = model.get_augmentation(flip=False if 'something' in dataset_name or 'jester' in dataset_name else True)
    
    path_frames = dict_conf['dataset']['path_frames']
    memory_size = dict_conf['memory']['memory_size']
    batch_size = dict_conf['batch_size']
    num_workers = dict_conf['num_workers']
    arch = dict_conf['feature_encoder']['type_clip_model']
    num_segments = dict_conf['feature_encoder']['num_segments']
    path_memory = dict_conf['memory']['path_memory']

    normalize = GroupNormalize(input_mean, input_std)
    data_length = 1
    
    train_transforms = torchvision.transforms.Compose([
        train_augmentation,
        Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
        normalize
    ])
    val_transforms = torchvision.transforms.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
        normalize,
    ])
    
    train_per_noise = dict_conf['dataset']['train_per_noise'] if 'train_per_noise' in dict_conf['dataset'] else 0
    val_per_noise = dict_conf['dataset']['val_per_noise'] if 'val_per_noise' in dict_conf['dataset'] else 0
    co_threshold = dict_conf['dataset']['co_threshold'] if 'co_threshold' in dict_conf['dataset'] else 0
    
    train_cilDatasetList = CILSetTask(data['train'], path_frames, memory_size, batch_size, shuffle=True, 
                                      num_workers=num_workers, num_frame_to_save = dict_conf['num_frame_to_save'], 
                                      is_activityNet = is_activityNet, per_noise = train_per_noise, co_threshold = co_threshold, 
                                      drop_last=True, pin_memory=True, num_segments=num_segments, new_length=data_length, 
                                      modality='RGB',transform=train_transforms, dense_sample=False, train_enable = True, name_dataset=dataset_name)
    
    val_cilDatasetList = CILSetTask(data['val'], path_frames, memory_size, batch_size, shuffle=False, 
                                    num_workers=num_workers, is_activityNet = is_activityNet, per_noise = val_per_noise, 
                                    co_threshold = co_threshold, pin_memory=True, num_frame_to_save = dict_conf['num_frame_to_save'], 
                                    num_segments=num_segments, new_length=data_length, modality='RGB', 
                                    transform=val_transforms, random_shift=False, dense_sample=False, train_enable = False, name_dataset=dataset_name)
    
    test_cilDatasetList = None
    if not is_activityNet:
        test_cilDatasetList = CILSetTask(data['test'], path_frames, memory_size, batch_size, shuffle=False, 
                                        num_workers=num_workers, is_activityNet = is_activityNet, per_noise = val_per_noise,
                                        co_threshold = co_threshold, pin_memory=True, num_frame_to_save = dict_conf['num_frame_to_save'], 
                                        num_segments=num_segments, new_length=data_length, modality='RGB', 
                                        transform=val_transforms, random_shift=False, dense_sample=False, train_enable = False, name_dataset=dataset_name)

    # define loss function (criterion) and optimizer
    cls_loss = nn.CrossEntropyLoss().to(device)
    textual_con_loss = TextualContrastiveLoss(dict_conf['batch_size'], device, temperature = dict_conf['temperature']).to(device)
    
    model.set_losses(cls_loss, textual_con_loss)

    if dict_conf['checkpoints']['train_mode']:
        model.add_num_classes(num_class)
        model.create_fc()
        train_loop(model, train_cilDatasetList, val_cilDatasetList, test_cilDatasetList)
        
def train_loop(model, train_cilDatasetList, val_cilDatasetList, test_cilDatasetList):
    iter_trainDataloader = iter(train_cilDatasetList)
    num_tasks = train_cilDatasetList.num_tasks
    path_memory = dict_conf['memory']['path_memory']
    
    for j in range(num_tasks):
        classes, data, train_loader_i, len_data, num_next_classes = next(iter_trainDataloader)
        if memory_size != 'ALL':
            m = memory_size // model.num_classes
        else:
            m = 'ALL'
        
        old_memory = model.memory
        model.add_samples_to_mem(data, m)
        train_loader_i_mem = None
        if m > 0 or m == 'ALL':
            data_mem = model.memory
            if model.num_training_phases > 1:
                data_mem = {**old_memory, **data}
            train_loader_i_mem = train_cilDatasetList.get_dataloader(data_mem, batch_size, None, False, False)
        
        if model.type_mod == 'None' and model.type_cls != 'Linear' and not model.enable_temporal_module:
            print('init validation')
            model.validate(val_cilDatasetList, j, train_loader_i.dataset.classes, 1, type_val = 'val', is_final = True)
            if not is_activityNet:
                with experiment.test():
                    total_acc_test,_ = model.validate(test_cilDatasetList, j, train_loader_i.dataset.classes, 1, type_val = 'test', is_final = True)
                    experiment.log_metric("Acc_task_{}".format(j+1), total_acc_test)
                    print('Test Accuracy: %d %%' % total_acc_test)
        else:
            print('init training')
            model.train_task(j, train_loader_i, train_loader_i_mem, val_cilDatasetList)
            if not is_activityNet:
                with experiment.test():
                    modulate_vid = True if model.type_mod == 'Prompt' else False
                    classes = train_loader_i_mem.dataset.classes if model.num_training_phases > 1 else train_loader_i.dataset.classes
                    total_acc_test = model.validate(test_cilDatasetList, j, classes, model.num_training_phases, type_val = 'test', is_final = True, modulate_vid = modulate_vid, modulate_txt = False)
                    total_acc_test = total_acc_test[0] if type(total_acc_test) == tuple else total_acc_test
                    experiment.log_metric("Acc_task_{}".format(j+1), total_acc_test)
                    print('Test Accuracy: %d %%' % total_acc_test)
        
        if model.num_training_phases == 1:
            train_cilDatasetList.memory = model.memory
        else:
            empty_memory = {}
            train_cilDatasetList.memory = empty_memory
        print('n_known_classes: ',len(model.memory))
        with open(path_memory, 'wb') as handle:
            pickle.dump(model.memory, handle)

        model.prepare_for_next_classes(num_next_classes)
        

if __name__ == '__main__':
    main()