from ast import Pass
from .CLIP.clip import clip
from .prompting_module import PromptModule
from .temporal_module import Temporal_Module
from .temporalShiftModule.ops.utils import AverageMeter, accuracy
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import random
from torchvision import transforms
from .temporalShiftModule.ops.transforms import *
import torch.nn.functional as F
import os

class BaselineModel(nn.Module):
    
    def __init__(self, device, conf, experiment):
        super(BaselineModel, self).__init__()
    
        self.conf = conf
        self.experiment = experiment
        self.device = device
        type_clip_model = conf['feature_encoder']['type_clip_model']
        type_mod = conf['modulation_module']['type_mod']
        self.type_mod = type_mod
        self.num_segments = conf['feature_encoder']['num_segments']
        self.use_text_template = conf['use_text_template']

        by_instance = conf['modulation_module']['mod_by_instance']
        
        L_tx_gn = conf['modulation_module']['prompt_module']['length_tex_gn']
        self.clip_model, _ = clip.load(type_clip_model, device=device, type_mod = type_mod, L_tx_gn = L_tx_gn)
        self.enable_temporal_module = conf['feature_encoder']['enable_temporal_module']
        if self.enable_temporal_module:
            self.temporal_module = Temporal_Module(device, conf['feature_encoder'])
            self.temporal_module = self.temporal_module.to(self.device)

            pytorch_total_params = sum(p.numel() for p in self.temporal_module.parameters() if p.requires_grad)
            print('num params of temp_module: ',pytorch_total_params)

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        width_vid = self.clip_model.vision_width
        width_txt = self.clip_model.transformer.width
        width_temp = conf['feature_encoder']['dim_model']
        self.width_temp = width_temp
        self.crop_size = self.clip_model.visual.input_resolution
        self.scale_size = self.crop_size * 256 // 224
        self.input_mean = [0.48145466, 0.4578275, 0.40821073]
        self.input_std = [0.26862954, 0.26130258, 0.27577711]

        self.num_classes = 0
        self.memory = {}
        self.type_cls = conf['type_loss']
        self.list_val_acc_ii = {'val': [], 'test': []}
        self.num_training_phases = conf['num_training_phases']

        self.prompt_module = None
        self.pre_pro_train_mode = False
        self.curr_pro_train_mode = False
        self.type_task = conf['type_task']
        self.training_phase_task_selector = conf['training_phase_task_selector']
        self.teacher_forcing = conf['modulation_module']['teacher_forcing']
        self.weights = conf['weights']
        self.val_weights = conf['val_weights']
        if type_mod == 'Prompt':
            conf_prompt_module = conf['modulation_module']['prompt_module']
            self.pre_pro_train_mode = conf_prompt_module['pre_pro_train_mode']
            self.curr_pro_train_mode = conf_prompt_module['curr_pro_train_mode']
            self.type_prompt = conf_prompt_module['type_prompt']
            self.prompt_module = PromptModule(conf_prompt_module, width_vid, width_temp, width_txt, self.type_prompt, self.num_segments, self.type_task, device)
        
    def add_num_classes(self, num_next_classes):
        self.num_classes+=num_next_classes

    def prepare_for_next_classes(self, num_next_classes):
        if num_next_classes != None:
            self.add_num_classes(num_next_classes)
            if self.type_cls == 'Linear':
                self.create_fc()
                print('Classifier augmented')

    def get_augmentation(self, flip=True):
        if flip:
            return transforms.Compose([GroupMultiScaleCrop(self.crop_size, [1, .875, .75, .66]),
                                                    GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return transforms.Compose([GroupMultiScaleCrop(self.crop_size, [1, .875, .75, .66])])
      
    def add_samples_to_mem(self, data, m):
        
        self.memory = {**self.memory, **data}
        for class_id, videos in self.memory.items():
            random.shuffle(videos)
            if m != 'ALL':
                self.memory[class_id] = videos[:m]
            else:
                self.memory[class_id] = videos

        for class_id, videos in self.memory.items():
            print('Memory... Class: {}, num videos: {}'.format(class_id, len(videos)))

    def create_fc(self):
        if self.type_cls == 'Linear':
            in_features = self.clip_model.output_dim
            self.classifier = nn.Linear(in_features, self.num_classes)
            self.classifier.to(self.device)

    def create_init_key(self, classes):
        list_curr_classes = []
        for cls in classes:
            if not cls in self.prompt_module.cls_to_task_id:
                list_curr_classes.append(cls)
        classes_emb = self.encode_labels(list_curr_classes, modulate = False)
        task_emb_init = torch.mean(classes_emb, dim=0, keepdim=True)
        return task_emb_init
        


    def prepare_trainining(self, classes, task_id, training_phase = 1, pre_pro_train_mode = False, curr_pro_train_mode = True):
        if self.num_training_phases == 1:
            if self.type_mod == 'Prompt' and self.type_prompt != 'general':
                self.prompt_module.prepare_task_prompt(classes, task_id)
                if self.prompt_module.L_sp_tk > 0:
                    self.prompt_module.set_train_mode_task_prompts(task_id, 'ViT', pre_pro_train_mode, curr_pro_train_mode)
                if self.prompt_module.L_sp_tk > 0:
                    self.prompt_module.set_train_mode_task_prompts(task_id, 'temp', pre_pro_train_mode, curr_pro_train_mode)
        else:
            if training_phase == 1 or training_phase == 3:
                if self.type_mod == 'Prompt' and self.type_prompt != 'general':
                    if self.prompt_module.L_sp_tk > 0:
                        self.prompt_module.set_train_mode_task_prompts(task_id, 'ViT', False, False)
                    if self.prompt_module.L_tp_tk > 0:
                        self.prompt_module.set_train_mode_task_prompts(task_id, 'temp', False, False)
                if self.enable_temporal_module:
                    for param in self.temporal_module.parameters():
                        param.requires_grad = True
            else:
                if self.type_mod == 'Prompt' and self.type_prompt != 'general':
                    self.prompt_module.prepare_task_prompt(classes, task_id)
                    if self.prompt_module.L_sp_tk > 0:
                        self.prompt_module.set_train_mode_task_prompts(task_id, 'ViT', pre_pro_train_mode, curr_pro_train_mode)
                    if self.prompt_module.L_tp_tk > 0:
                        self.prompt_module.set_train_mode_task_prompts(task_id, 'temp', pre_pro_train_mode, curr_pro_train_mode)
                if self.enable_temporal_module:
                    for param in self.temporal_module.parameters():
                        param.requires_grad = False

    def get_optimizer(self, training_phase = 1):
        print('here4 optimizer')
        params_out = []
        if self.num_training_phases == 1:
            print('here 1 phases optimizer')
            if self.type_mod == 'Prompt':
                for p in self.prompt_module.parameters():
                    print('p: ',p)
                    params_out.append(p)

            if self.type_cls == 'Linear':
                print('adding linear cls parameters')
                for p in self.classifier.parameters():
                    params_out.append(p)
            
            if self.enable_temporal_module:
                print('adding temp parameters')
                for p in self.temporal_module.parameters():
                    params_out.append(p)
        else:
            print('here 2 phases optimizer')
            if training_phase == 1 or training_phase == 3:
                print('here first phase')
                if self.enable_temporal_module:
                    print('adding temp parameters')
                    for p in self.temporal_module.parameters():
                        print('p: ',p)
                        params_out.append(p)
                if self.type_cls == 'Linear':
                    print('adding linear cls parameters')
                    for p in self.classifier.parameters():
                        print('p: ',p)
                        params_out.append(p)
            else:
                print('here second phase')
                if self.type_mod == 'Prompt':
                    print('adding prompting parameters')
                    for p in self.prompt_module.parameters():
                        print('p: ',p)
                        params_out.append(p)
                if self.type_cls == 'Linear':
                    print('adding temp parameters')
                    for p in self.classifier.parameters():
                        params_out.append(p)

        self.optimizer = torch.optim.SGD(params_out, lr=self.conf['lr'])
    
    def set_losses(self, cls_loss, text_con_loss = None):
        self.cls_loss = cls_loss
        self.text_con_loss = text_con_loss

    def forward_selector(self, videos, classes, modulate_txt):
        videos = videos.to(self.device)
        videos = videos.view((-1, 3) + videos.size()[-2:])
        videos_features = self.clip_model.encode_image(videos)
        videos_features = videos_features.view(videos_features.size(0) // self.num_segments, self.num_segments, -1)

        if self.enable_temporal_module:
            video_emb = self.temporal_module(videos_features)
        else:
            video_emb = torch.mean(videos_features, dim=1)

        
        dict_saved_classes = self.prompt_module.cls_to_task_id
        classes = list(classes)
        learned_classes = []
        for cls in classes:
            if cls in dict_saved_classes or cls.replace(' ', '') in dict_saved_classes:
                learned_classes.append(cls)

        classes_emb = self.encode_labels(learned_classes, modulate_txt)
        video_emb = video_emb.unsqueeze(dim=1)
        video_emb = video_emb.expand(video_emb.size(0), classes_emb.size(0), video_emb.size(2))
        classes_emb = classes_emb.unsqueeze(dim=0)
        classes_emb = classes_emb.expand(video_emb.size(0), classes_emb.size(1), classes_emb.size(2))
        output = F.cosine_similarity(video_emb, classes_emb, dim=2)

        _, class_ids = torch.max(output, 1)
        dict_saved_classes = self.prompt_module.cls_to_task_id
        task_ids_preds = []
        for i in class_ids:
            cls = classes[i]
            if cls in dict_saved_classes or cls.replace(' ', '') in dict_saved_classes:
                cls = cls if cls in dict_saved_classes else cls.replace(' ', '')
                task_ids_preds.append(dict_saved_classes[cls])
        task_ids_preds = torch.LongTensor(task_ids_preds).to(self.device)

        return task_ids_preds


    def encode_videos(self, videos, modulate = False, type_task = None):
        videos = videos.to(self.device)
        videos = videos.view((-1, 3) + videos.size()[-2:])
        if self.type_mod == 'Prompt' and modulate and type_task != None:
            videos_features = self.clip_model.encode_image(videos, promptModule = self.prompt_module, type_task = type_task)
        else:
            videos_features = self.clip_model.encode_image(videos)
        videos_features = videos_features.view(videos_features.size(0) // self.num_segments, self.num_segments, -1)
        if self.enable_temporal_module:
            if self.type_mod == 'Prompt' and modulate and type_task != None:
                context_emb = self.temporal_module(videos_features, promptModule = self.prompt_module, type_task = type_task)
            else:
                context_emb = self.temporal_module(videos_features)
        else:
            context_emb = torch.mean(videos_features, dim=1)
         
        return context_emb
    
    
    def encode_labels(self, textual_descrips, modulate = False, type_task = None):
        if self.use_text_template:
            textual_descrips = ['a photo of a '+cls_txt.lower() for cls_txt in textual_descrips]
        text_tokens = clip.tokenize(textual_descrips).to(self.device)
        if self.type_mod == 'Prompt' and modulate and type_task != None:
            text_features = self.clip_model.encode_text(text_tokens, promptModule = self.prompt_module, type_task = type_task)
        else:
            text_features = self.clip_model.encode_text(text_tokens)
        return text_features

    def set_train_mode(self, task_id, training_phase, pre_pro_train_mode, curr_pro_train_mode):
        if self.num_training_phases == 1:
            if self.type_mod == 'Prompt':
                if self.type_prompt != 'general':
                    if self.prompt_module.L_sp_tk > 0:
                        self.prompt_module.set_train_mode_task_prompts(task_id, 'ViT', pre_pro_train_mode, curr_pro_train_mode)
                    if self.prompt_module.L_tp_tk > 0:
                        self.prompt_module.set_train_mode_task_prompts(task_id, 'temp', pre_pro_train_mode, curr_pro_train_mode)
                else:
                    self.prompt_module.train()
            if self.enable_temporal_module:
                self.temporal_module.train()           
        else:
            if training_phase == 1 or training_phase == 3:
                if self.type_mod == 'Prompt':
                    if self.type_prompt != 'general':
                        if self.prompt_module.L_sp_tk > 0:
                            self.prompt_module.set_train_mode_task_prompts(task_id, 'ViT', False, False)
                        if self.prompt_module.L_tp_tk > 0:
                            self.prompt_module.set_train_mode_task_prompts(task_id, 'temp', False, False)
                    else:
                        self.prompt_module.eval()
                if self.enable_temporal_module:
                    self.temporal_module.train()
            else:
                if self.type_mod == 'Prompt':
                    if self.type_prompt != 'general':
                        if self.prompt_module.L_sp_tk > 0:
                            self.prompt_module.set_train_mode_task_prompts(task_id, 'ViT', pre_pro_train_mode, curr_pro_train_mode)
                        if self.prompt_module.L_tp_tk > 0:
                            self.prompt_module.set_train_mode_task_prompts(task_id, 'temp', pre_pro_train_mode, curr_pro_train_mode)
                    else:
                        self.prompt_module.train()
                if self.enable_temporal_module:
                    self.temporal_module.eval()

        if self.type_cls == 'Linear':
            self.classifier.train()
        
        self.clip_model.eval()


    def set_eval_mode(self):
        if self.type_mod == 'Prompt':
            self.prompt_module.eval()
        if self.type_cls == 'Linear':
            self.classifier.eval()
        if self.enable_temporal_module:
            self.temporal_module.eval()
        
        self.clip_model.eval()
        
    def count_accuracy(self, video_emb, classes, labels, modulate_txt):
        classes = list(classes)
        classes_emb = self.encode_labels(classes, modulate_txt)
        video_emb = video_emb.unsqueeze(dim=1)
        video_emb = video_emb.expand(video_emb.size(0), classes_emb.size(0), video_emb.size(2))
        classes_emb = classes_emb.unsqueeze(dim=0)
        classes_emb = classes_emb.expand(video_emb.size(0), classes_emb.size(1), classes_emb.size(2))
        output = F.cosine_similarity(video_emb, classes_emb, dim=2)
        acc = accuracy(output.data, labels, topk=(1,))[0]
        acc = acc.item()
        return acc

    def validate_task(self, val_cilDatasetList, current_task_id, modulate_vid, modulate_txt, classes, training_phase, is_final, type_val):
        print('Init val task')
        top1 = AverageMeter()
        val_loader, _ = val_cilDatasetList.get_validation_task(current_task_id, curr_classes=True)
        self.set_eval_mode()
        BWF = AverageMeter()
        with torch.no_grad():
            for _, _, videos, _, labels, text_descrip in val_loader:
                labels = labels.to(self.device)
                # compute output
                with autocast():
                    acc_val, _, _ = self.forward_pass_video(False, modulate_vid, modulate_txt, videos, labels, text_descrip, classes, training_phase, is_train=False)
                    
                top1.update(acc_val, videos.size(0))
            
            if is_final and current_task_id == 0:
                self.experiment.log_metric("Acc_task_{}".format(current_task_id+1), top1.avg, step=current_task_id+1)
                self.list_val_acc_ii[type_val].append(top1.avg)

                self.experiment.log_metric("Total_Acc_Per_task", top1.avg, step=current_task_id+1)
                self.experiment.log_metric("Total_BWF_Per_task", BWF.avg, step=current_task_id+1)   

        return top1.avg, None
    
    def famework_validation_task(self, val_cilDatasetList, current_task_id, classes, training_phase, type_val = 'val', is_final= False, modulate_vid= False, modulate_txt = False, split_batch = False):
        if self.num_training_phases == 1 or (self.num_training_phases > 1 and training_phase == 3):
            print('here total val, training phase: ',training_phase)
            return self.validate(val_cilDatasetList, current_task_id, classes, training_phase, type_val, is_final, modulate_vid, modulate_txt, split_batch)
        else:
            print('here val task, training phase: ',training_phase)
            return self.validate_task(val_cilDatasetList, current_task_id, modulate_vid, modulate_txt, classes, training_phase, is_final, type_val)


    def validate(self, val_cilDatasetList, current_task_id, classes, training_phase, type_val = 'val', is_final= False, modulate_vid= False, modulate_txt = False, split_batch = False):
        print('Init val')
        top1 = AverageMeter()
        total_acc = AverageMeter()
        top1_aux = AverageMeter()
        total_acc_aux = AverageMeter()
        val_loaders_list, _ = val_cilDatasetList.get_valSet_by_taskNum(current_task_id+1)
        BWF = AverageMeter()
        # switch to evaluate mode
        self.set_eval_mode()
        
        with torch.no_grad():
            for n_task, (val_loader, num_classes) in enumerate(val_loaders_list):
                for _, _, videos, _, labels, text_descrip in val_loader:
                    labels = labels.to(self.device)
                    # compute output
                    with autocast():
                        acc_val, _, aux_acc_val = self.forward_pass_video(split_batch, modulate_vid, modulate_txt, videos, labels, text_descrip, classes, training_phase, is_train=False)
                        
                    top1.update(acc_val, videos.size(0))
                    if self.type_mod == 'Prompt' and self.type_task == 'CIL' and self.type_prompt != 'general' and training_phase == self.training_phase_task_selector and len(self.prompt_module.cls_to_task_id)>0:
                        top1_aux.update(aux_acc_val, videos.size(0))
                
                total_acc.update(top1.avg, num_classes)
                print('Train... task : {}, acc with classifier: {}'.format(n_task, top1.avg))
                if self.type_mod == 'Prompt' and self.type_task == 'CIL' and self.type_prompt != 'general' and training_phase == self.training_phase_task_selector and len(self.prompt_module.cls_to_task_id)>0:
                    total_acc_aux.update(top1_aux.avg, num_classes)
                    print('Train... task : {}, aux acc with classifier: {}'.format(n_task, top1_aux.avg))
                if is_final:
                    self.experiment.log_metric("Acc_task_{}".format(n_task+1), top1.avg, step=current_task_id+1)
                    if n_task == current_task_id:
                        self.list_val_acc_ii[type_val].append(top1.avg)
                    elif n_task < current_task_id:
                        forgetting = self.list_val_acc_ii[type_val][n_task] - top1.avg
                        BWF.update(forgetting, num_classes)
                
                top1.reset()
                if self.type_mod == 'Prompt' and self.type_task == 'CIL' and self.type_prompt != 'general' and training_phase == self.training_phase_task_selector and len(self.prompt_module.cls_to_task_id)>0:
                    top1_aux.reset()
        output = ('Pre Testing Results: Pre_Acc {total_acc.avg:.3f}'
                  .format(total_acc=total_acc))
        
        if is_final:
            self.experiment.log_metric("Total_Acc_Per_task", total_acc.avg, step=current_task_id+1)
            self.experiment.log_metric("Total_BWF_Per_task", BWF.avg, step=current_task_id+1)
        print(output)
        avg_total_acc_aux = None
        if self.type_mod == 'Prompt' and self.type_task == 'CIL' and self.type_prompt != 'general' and training_phase == self.training_phase_task_selector and len(self.prompt_module.cls_to_task_id)>0:
            print('Pre Testing Results: Pre_Acc_AUX {total_acc_aux.avg:.3f}'.format(total_acc_aux=total_acc_aux))
            avg_total_acc_aux = total_acc_aux.avg
        return total_acc.avg, avg_total_acc_aux

    def load_best_checkpoint(self, path_model, current_task):
        if os.path.exists(path_model):
            checkpoint_dict = torch.load(path_model)
            task_to_load = checkpoint_dict['current_task']
            epoch_to_load = checkpoint_dict['current_epoch']
            if task_to_load == current_task:
                print('Loading best checkpoint ... epoch: {}, task: {}'.format(epoch_to_load, task_to_load))
                if self.type_mod == 'Prompt':
                    self.prompt_module.load_state_dict(checkpoint_dict['state_dict_prompt'])
                if self.type_cls == 'Linear':
                    self.classifier.load_state_dict(checkpoint_dict['state_dict_classifier'])
                if self.enable_temporal_module:
                    self.temporal_module.load_state_dict(checkpoint_dict['state_dict_temporal_module'])

    def get_task_label(self, text_descriptions):
        dict_saved_classes = self.prompt_module.cls_to_task_id
        task_ids = []
        for cls in text_descriptions:
            if cls in dict_saved_classes or cls.replace(' ', '') in dict_saved_classes:
                cls = cls if cls in dict_saved_classes else cls.replace(' ', '')
                task_ids.append(dict_saved_classes[cls])
        task_ids = torch.LongTensor(task_ids).to(self.device)
        return task_ids

    def forward_pass_video(self, split_batch, modulate_vid, modulate_txt, videos, labels, text_descrip, classes, training_phase, curr_task_id=None, is_train = True):
        if split_batch and self.type_mod == 'Prompt' and len(self.prompt_module.cls_to_task_id)>0:
            batch_to_prompt, label_prompt, text_decrip_prompt, batch_novel_cls, label_cls, text_descrip_cls = self.split_batch(videos, labels, text_descrip, curr_task_id)
            if batch_to_prompt != None:
                if self.teacher_forcing or self.type_task == 'TIL' or self.type_prompt == 'general':
                    self.prompt_module.set_current_video_classes(text_decrip_prompt)
                    video_emb_prompt = self.encode_videos(batch_to_prompt, True, 'TIL')
                else:
                    task_ids_preds = self.forward_selector(batch_to_prompt, classes, modulate_txt)
                    self.prompt_module.set_current_task_ids(task_ids_preds)
                    video_emb_prompt = self.encode_videos(batch_to_prompt, True, 'CIL')
                video_emb_prompt_wo_mod = self.encode_videos(batch_to_prompt, False)
                video_emb_prompt = torch.cat([video_emb_prompt, video_emb_prompt_wo_mod], dim=0)
                video_emb = video_emb_prompt
                label_prompt = torch.cat([label_prompt, label_prompt], dim=0)
                labels = label_prompt
                text_decrip_prompt.extend(text_decrip_prompt)
                text_descrip = text_decrip_prompt
                batch_to_prompt = torch.cat([batch_to_prompt, batch_to_prompt], dim=0)
                videos = batch_to_prompt
            if batch_novel_cls != None:
                video_emb_cls = self.encode_videos(batch_novel_cls, False)
                video_emb = video_emb_cls
            if batch_novel_cls != None and batch_to_prompt != None:
                video_emb = torch.cat([video_emb_prompt, video_emb_cls], dim = 0)
                labels = torch.cat([label_prompt, label_cls], dim = 0)
                text_decrip_prompt.extend(text_descrip_cls)
                idx = list(range(video_emb.size(0)))
                random.shuffle(idx)
                text_descrip = [text_decrip_prompt[i] for i in idx]
                idx = torch.LongTensor(idx)
                video_emb = video_emb[idx]
                labels = labels[idx]

                batch_videos = torch.cat([batch_to_prompt, batch_novel_cls], dim = 0)
                videos = batch_videos[idx]
        else:
            if self.type_mod == 'Prompt':
                enable_cil = self.num_training_phases == 1 or (self.num_training_phases > 1 and training_phase == 3)
                if self.type_task == 'CIL' and len(self.prompt_module.cls_to_task_id)>0 and enable_cil:
                    if modulate_vid and self.type_prompt != 'general':
                        task_ids_pred = self.forward_selector(videos, classes, modulate_txt)
                        self.prompt_module.set_current_task_ids(task_ids_pred)
                    video_emb = self.encode_videos(videos, modulate_vid, 'CIL')
                else:
                    self.prompt_module.set_current_video_classes(text_descrip)
                    video_emb = self.encode_videos(videos, modulate_vid, 'TIL')
            else:
                video_emb = self.encode_videos(videos, modulate_vid)
        
        aux_acc_train = None
        if self.type_mod == 'Prompt' and self.type_task == 'CIL' and self.type_prompt != 'general' and training_phase == self.training_phase_task_selector and len(self.prompt_module.cls_to_task_id)>0:
            gt_tasks_ids = self.get_task_label(text_descrip)
            task_ids_pred = self.forward_selector(videos, classes, modulate_txt)
            aux_acc_train = torch.eq(task_ids_pred, gt_tasks_ids)
            aux_acc_train = (aux_acc_train.float().sum(0))*100/gt_tasks_ids.size(0)
            aux_acc_train = aux_acc_train.item()

        if self.type_cls == 'Linear':
            preds = self.classifier(video_emb)
            loss = self.cls_loss(preds, labels)
            acc_train = accuracy(preds.data, labels, topk=(1,))[0]
            acc_train = acc_train.item()
        else:
            text_emb = self.encode_labels(text_descrip, modulate_txt)
            # normalized features
            video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.clip_model.logit_scale.exp()
            logits_per_image = logit_scale * video_emb @ text_emb.t()
            logits_per_text = logits_per_image.t()
            ground_truth = torch.arange(len(videos),dtype=torch.long,device=self.device)
            loss = (self.cls_loss(logits_per_image,ground_truth) + self.cls_loss(logits_per_text,ground_truth))/2
            acc_train = self.count_accuracy(video_emb, classes, labels, modulate_txt)

        return acc_train, loss, aux_acc_train

    def split_batch(self, x, labels, text_description, curr_task_id):
        batch_to_prompt, label_prompt, text_decrip_prompt = [], [], []
        batch_novel_cls, label_cls, text_descrip_cls = [], [], []
        dict_saved_classes = self.prompt_module.cls_to_task_id
        for i in range(len(text_description)):
            cls = text_description[i]
            if cls in dict_saved_classes or cls.replace(' ', '') in dict_saved_classes:
                batch_to_prompt.append(x[i])
                label_prompt.append(labels[i])
                text_decrip_prompt.append(text_description[i])
            else:
                batch_novel_cls.append(x[i])
                label_cls.append(labels[i])
                text_descrip_cls.append(text_description[i])

        batch_to_prompt = torch.stack(batch_to_prompt, dim=0) if len(batch_to_prompt) > 0 else None
        label_prompt = torch.stack(label_prompt, dim=0) if len(label_prompt) > 0 else None

        batch_novel_cls = torch.stack(batch_novel_cls, dim=0) if len(batch_novel_cls) > 0 else None
        label_cls = torch.stack(label_cls, dim=0) if len(label_cls) > 0 else None
        return batch_to_prompt, label_prompt, text_decrip_prompt, batch_novel_cls, label_cls, text_descrip_cls
    
    def save_checkpoint(self, path_model, acc_val, epoch, task_id, is_best):
        if is_best and (self.type_mod == 'Prompt' or self.type_cls == 'Linear'):
            print('Saving ... ')
            dict_to_save = {'accuracy': acc_val, 'current_epoch': epoch, 
                            'current_task': task_id, 'optimizer': self.optimizer.state_dict()}
            if self.type_mod == 'Prompt':
                dict_to_save['state_dict_prompt'] = self.prompt_module.state_dict()
            if self.type_cls == 'Linear':
                dict_to_save['state_dict_classifier'] = self.classifier.state_dict()
            if self.enable_temporal_module:
                dict_to_save['state_dict_temporal_module'] = self.temporal_module.state_dict()
            torch.save(dict_to_save, path_model)
            print("Save Best Networks for task: {}, epoch: {}".format(dict_to_save['current_task'] + 1, 
                                                                 dict_to_save['current_epoch'] + 1), flush=True)
    
    def train_phase(self, task_id, training_phase, pre_pro_train_mode, curr_pro_train_mode, train_dataloader_cil, val_cilDatasetList, modulate_vid, modulate_txt, split_batch, is_final):

        eval_freq = self.conf['checkpoints']['eval_freq']
        path_model = self.conf['checkpoints']['path_model']
        num_epochs = self.conf['epochs']
        best_acc_val = 0
        
        self.prepare_trainining(train_dataloader_cil.dataset.classes, task_id, training_phase, pre_pro_train_mode, curr_pro_train_mode)
        self.get_optimizer(training_phase)
        self.optimizer.zero_grad()
        with self.experiment.train():
            for epoch in range(num_epochs):
                self.set_train_mode(task_id, training_phase, pre_pro_train_mode, curr_pro_train_mode)
                acc_Avg = AverageMeter()
                loss_Avg = AverageMeter()
                aux_acc_Avg = AverageMeter()
                for i, (indices, _, videos, _, labels, text_descrip) in enumerate(train_dataloader_cil):
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    with autocast():
                        acc_train, loss, aux_acc_train = self.forward_pass_video(split_batch, modulate_vid, modulate_txt, videos, labels, text_descrip, train_dataloader_cil.dataset.classes, training_phase, curr_task_id=task_id, is_train=True)
                        
                    if self.type_mod == 'Prompt' and self.type_task == 'CIL' and self.type_prompt != 'general' and training_phase == self.training_phase_task_selector and len(self.prompt_module.cls_to_task_id)>0:
                        self.experiment.log_metric("Aux_Acc_task_{}".format(task_id+1), aux_acc_train)
                        aux_acc_Avg.update(aux_acc_train, videos.size(0))

                    loss.backward()
                    self.optimizer.step()
                    
                    self.experiment.log_metric("Acc_task_{}".format(task_id+1), acc_train)
                    self.experiment.log_metric("Loss_task_{}".format(task_id+1), loss.item())
                    loss_Avg.update(loss.item(), videos.size(0))
                    acc_Avg.update(acc_train, videos.size(0))
                
                    if (i+1) % 2 == 0:
                        print('Epoch [%d/%d], Loss: %.4f' 
                            %(epoch+1, num_epochs, loss.item()))
                    
                self.experiment.log_metric("Epoch_Acc_task_{}".format(task_id+1), acc_Avg.avg)
                self.experiment.log_metric("Epoch_Loss_task_{}".format(task_id+1), loss_Avg.avg)

                if self.type_mod == 'Prompt' and self.type_task == 'CIL' and self.type_prompt != 'general' and training_phase == self.training_phase_task_selector:
                    self.experiment.log_metric("Epoch_Aux_Acc_task_{}".format(task_id+1), aux_acc_Avg.avg)
                    
                if (epoch + 1) % eval_freq == 0 or epoch == num_epochs - 1:
                    with self.experiment.validate():  
                        classes = train_dataloader_cil.dataset.classes
                        acc_val, aux_acc_val = self.famework_validation_task(val_cilDatasetList, task_id, classes, training_phase, 'val', False, modulate_vid, modulate_txt, False)
                        self.experiment.log_metric("Acc_at_task_{}".format(task_id+1), acc_val)
                        total_acc_val = self.val_weights[0]*acc_val + self.val_weights[1]*aux_acc_val if aux_acc_val != None else acc_val
                        is_best = total_acc_val >= best_acc_val
                        best_acc_val = max(total_acc_val, best_acc_val)
                        output_best = 'Best Avg Pre Acc Val@1: %.3f\n' % (best_acc_val)
                        print(output_best)
                        self.save_checkpoint(path_model, acc_val, epoch, task_id, is_best)
                        if epoch == num_epochs - 1:
                            if not is_best:
                                self.load_best_checkpoint(path_model, task_id)
                            self.famework_validation_task(val_cilDatasetList, task_id, classes, training_phase, 'val', is_final, modulate_vid, modulate_txt, False)
    
    def train_task(self, task_id, train_dataloader_cil, train_dataloader_men, val_cilDatasetList):
        print('Task # {}'.format(task_id+1))
        if self.num_training_phases == 1:
            print('Training phase #1')
            self.train_phase(task_id, 1, self.pre_pro_train_mode, self.curr_pro_train_mode, train_dataloader_cil, val_cilDatasetList, modulate_vid = True, modulate_txt=False, split_batch = False, is_final = True)
        else:
            print('Training phase #1')
            self.train_phase(task_id, 1, False, False, train_dataloader_cil, val_cilDatasetList, modulate_vid = False, modulate_txt = False, split_batch = False, is_final = False)
            print('Training phase #2')
            is_final = True if task_id == 0 else False
            self.train_phase(task_id, 2, self.pre_pro_train_mode, self.curr_pro_train_mode, train_dataloader_cil, val_cilDatasetList, modulate_vid = True, modulate_txt = False, split_batch = False, is_final = is_final)
            if task_id > 0:
                print('Training phase #3')
                self.train_phase(task_id, 3, False, False, train_dataloader_men, val_cilDatasetList, modulate_vid = True, modulate_txt = False, split_batch = True, is_final = True)