from gc import freeze
import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptModule(nn.Module):
    def __init__(self, conf_prompt, vid_width, temp_width, text_width, type_prompt, num_segments, type_task, device):
        super(PromptModule, self).__init__()
        
        print('type_prompt: ', type_prompt)
        self.type_prompt = type_prompt
        self.device = device
        self.vid_width = vid_width
        self.temp_width = temp_width
        self.num_segments = num_segments
        self.type_task = type_task
        if type_prompt == 'general':
            self.num_sel_prompts = 1
            self.idx_P_gn = {}
            self.L_tx_gn = conf_prompt['length_tex_gn']
            self.L_sp_gn = conf_prompt['length_sp_gn']
            self.L_tp_gn = conf_prompt['length_tp_gn']
            
            if self.L_tx_gn > 0:
                self.P_gn_txt = nn.Embedding(self.L_tx_gn, text_width).to(device)
                nn.init.normal_(self.P_gn_txt.weight, std=0.02)
                idx_P_tx_gn = torch.tensor(list(range(self.L_tx_gn))).to(device)
                self.idx_P_gn['txt'] = idx_P_tx_gn
            if self.L_sp_gn > 0:
                self.P_gn_ViT = nn.Embedding(self.L_sp_gn, vid_width).to(device)
                nn.init.normal_(self.P_gn_ViT.weight, std=0.02)
                idx_P_sp_gn = torch.tensor(list(range(self.L_sp_gn))).to(device)
                self.idx_P_gn['ViT'] = idx_P_sp_gn
            if self.L_tp_gn > 0:
                self.P_gn_temp = nn.Embedding(self.L_tp_gn, temp_width).to(device)
                nn.init.normal_(self.P_gn_temp.weight, std=0.02)
                idx_P_tp_gn = torch.tensor(list(range(self.L_tp_gn))).to(device)
                self.idx_P_gn['temp'] = idx_P_tp_gn
        elif type_prompt == 'learning_to_prompt':
            self.num_sel_prompts = conf_prompt['num_sel_prompts']
            self.num_prompts = conf_prompt['num_prompts']
            self.L_sp_tk = conf_prompt['length_sp_task']

            for j in range(self.num_prompts):
                P_sp_j = self.create_task_prompt(self.L_sp_tk, self.vid_width)
                setattr(self, 'P_ViT_{}'.format(j), P_sp_j)
                K_ViT_i = nn.Embedding(1, temp_width).to(device)
                nn.init.normal_(K_ViT_i.weight, std=0.02)
                setattr(self, 'K_ViT_{}'.format(j), K_ViT_i)
            
            idx_P_ViT = torch.tensor(list(range(self.L_sp_tk))).to(device)
            idx_K_ViT = torch.tensor(list(range(1))).to(device)
            self.idx_P = {'P_ViT': idx_P_ViT, 'K_ViT': idx_K_ViT} 
            self.val_sim_loss = torch.zeros(1).to(device)
            self.frequency = torch.zeros(self.num_prompts).to(device)
            self.freq_acc = torch.zeros(self.num_prompts).to(device)

        else:
            self.num_sel_prompts = conf_prompt['num_sel_prompts'] if 'num_sel_prompts' in conf_prompt else 1
            self.L_sp_tk = conf_prompt['length_sp_task']
            self.L_tp_tk = conf_prompt['length_tp_task']
            self.L_k_tk = 1
            self.cls_to_task_id = {}
            self.task_sel = None
            # by tasks

    def set_current_video_classes(self, text_description):
        self.text_description = text_description
    
    def set_video_emb_wo_prompt(self, video_emb_wo_prompt, is_train):
        self.video_emb_wo_prompt = video_emb_wo_prompt
        self.is_train = is_train

    def set_current_task_ids(self, task_ids_pred):
        self.task_ids_pred = task_ids_pred

    def train_mode_parameters(self, name, train_mode):
        embs = getattr(self, name)
        for param in embs.parameters():
            param.requires_grad = train_mode  
        if train_mode:
            embs.train()
        else:
            embs.eval()
        setattr(self, name, embs)

    def set_train_mode_task_prompts(self, current_task_id, source, pre_pro_train_mode = False, curr_pro_train_mode = True):
        tasks_id_saved = set(self.cls_to_task_id.values())
        if bool(tasks_id_saved):
            for task_id in tasks_id_saved:
                train_mode = pre_pro_train_mode if task_id != current_task_id else curr_pro_train_mode
                name_emb = 'P_tk_{}_{}'.format(source, task_id)
                self.train_mode_parameters(name_emb, train_mode)       
                
    def create_task_prompt(self, L_tk, dim, initialization=None):
        if initialization == None:
            P_tk = nn.Embedding(L_tk, dim).to(self.device)
            nn.init.normal_(P_tk.weight, std=0.02)
        else:
            P_tk = nn.Embedding.from_pretrained(initialization, freeze=False)
        return P_tk

    def prepare_task_prompt(self, classes, task_id, init_emb=None):
        tasks_id_saved = set(self.cls_to_task_id.values())
        if not task_id in tasks_id_saved:
            if self.L_sp_tk > 0:
                P_sp_tk = self.create_task_prompt(self.num_sel_prompts*self.L_sp_tk, self.vid_width)
                P_sp_tk_total_params = sum(p.numel() for p in P_sp_tk.parameters() if p.requires_grad)
                print('num params of P_sp_tk: ',P_sp_tk_total_params)
                setattr(self, 'P_tk_ViT_{}'.format(task_id), P_sp_tk)
            if self.L_tp_tk > 0:
                P_tp_tk = self.create_task_prompt(self.num_sel_prompts*self.L_tp_tk, self.temp_width)
                P_tp_tk_total_params = sum(p.numel() for p in P_tp_tk.parameters() if p.requires_grad)
                print('num params of P_tp_tk: ',P_tp_tk_total_params)
                setattr(self, 'P_tk_temp_{}'.format(task_id), P_tp_tk)
            if self.L_sp_tk > 0 or self.L_tp_tk > 0:
                for cls in classes: 
                    if not cls in self.cls_to_task_id:
                        self.cls_to_task_id[cls] = task_id

    def select_prompts_and_keys_to_train(self):
        index_not_sel = (self.freq_acc == 0).nonzero()
        for ind in index_not_sel:
            j = ind[0].item()
            prompt = getattr(self, 'P_ViT_{}'.format(j))
            key = getattr(self, 'K_ViT_{}'.format(j))
            for param in prompt.parameters():
                param.requires_grad = False  
            setattr(self, 'P_ViT_{}'.format(j), prompt)
            for param in key.parameters():
                param.requires_grad = False  
            setattr(self, 'K_ViT_{}'.format(j), key)

    def compute_distance_vid_keys(self, vid_emb, keys):
        if len(keys.size()) == 3:
            vid_emb = vid_emb.unsqueeze(dim=1)
            vid_emb = vid_emb.expand(vid_emb.size(0), keys.size(1), vid_emb.size(2))
        else:
            vid_emb = vid_emb.unsqueeze(dim=1)
            vid_emb = vid_emb.expand(vid_emb.size(0), keys.size(0), vid_emb.size(2))
            
            keys = keys.unsqueeze(dim=0)
            keys = keys.expand(vid_emb.size(0), keys.size(1), keys.size(2))  
            
        output_sel = 1 - F.cosine_similarity(vid_emb, keys, dim=2)
        return output_sel


    def get_LP_selector(self, source, x):
        if source == 'ViT':
            x = x.view(x.size(0)//self.num_segments, self.num_segments, x.size(1), x.size(2))
        
        prompts = []
        keys = []
        for j in range(self.num_prompts):
            prompt = getattr(self, 'P_{}_{}'.format(source, j))
            key = getattr(self, 'K_{}_{}'.format(source, j))
            idx = self.idx_P['P_{}'.format(source)]
            K_idx = self.idx_P['K_{}'.format(source)]
            prompt = prompt(idx)
            prompts.append(prompt)
            key = key(K_idx).squeeze(dim=0)
            keys.append(key)
        prompts = torch.stack(prompts, dim=0)
        keys = torch.stack(keys, dim = 0)

        video_emb_wo_prompt = F.normalize(self.video_emb_wo_prompt,dim=-1)
        keys = F.normalize(keys,dim=-1)
        output_sel = self.compute_distance_vid_keys(video_emb_wo_prompt, keys)

        if torch.sum(self.frequency) != 0 and self.is_train: 
            freq = self.frequency / torch.sum(self.frequency)
            freq = freq.unsqueeze(dim=0)
            freq = freq.expand(video_emb_wo_prompt.size(0), freq.size(1))
            output_reg = output_sel * freq
            _, sel_index = torch.topk(output_reg, self.num_sel_prompts, dim = 1, largest = False)
        else:
            _, sel_index = torch.topk(output_sel, self.num_sel_prompts, dim = 1, largest = False)

        batch_size = x.size(0)
        list_prompting_elem = []
        sel_keys = []
        for i in range(batch_size):
            elem = x[i]
            sel_index_elem = sel_index[i,:]
            self.freq_acc[sel_index_elem]+=1
            sel_prompts = prompts[sel_index_elem,:,:]
            sel_prompts = sel_prompts.view(-1, sel_prompts.size(2))
            if source == 'ViT': 
                sel_prompts = sel_prompts.unsqueeze(dim=0)
                sel_prompts = sel_prompts.expand(elem.size(0), sel_prompts.size(1), sel_prompts.size(2))
                elem = torch.cat([sel_prompts, elem], dim = 1)
            else:
                elem = torch.cat([sel_prompts, elem], dim = 0)
            list_prompting_elem.append(elem)
            sel_keys.append(keys[sel_index_elem,:])
        prompting_elem = torch.stack(list_prompting_elem, dim=0)
        sel_keys_by_elem = torch.stack(sel_keys, dim=0)
        dis_loss = self.compute_distance_vid_keys(video_emb_wo_prompt, sel_keys_by_elem)
        self.val_sim_loss = torch.mean(torch.sum(dis_loss, dim=1), dim=0) 

        if source == 'ViT':
            prompting_elem = prompting_elem.view(-1, prompting_elem.size(2), prompting_elem.size(3))

        return prompting_elem

    def get_task_selector(self, source, x):

        tasks_ids = self.task_ids_pred

        list_prompting_elem = []
        if source == 'ViT':
            x = x.view(x.size(0)//self.num_segments, self.num_segments, x.size(1), x.size(2))
        for i, task_id in enumerate(tasks_ids):
            L_tk = self.num_sel_prompts*self.L_sp_tk if source == 'ViT' else self.num_sel_prompts*self.L_tp_tk
            idx = torch.tensor(list(range(L_tk))).to(self.device)
            prompt_emb = getattr(self, 'P_tk_{}_{}'.format(source, task_id))(idx)
            elem = x[i]
            if source == 'ViT': 
                prompt_emb = prompt_emb.unsqueeze(dim=0)
                prompt_emb = prompt_emb.expand(elem.size(0), prompt_emb.size(1), prompt_emb.size(2))
                elem = torch.cat([prompt_emb, elem], dim = 1)
            else:
                elem = torch.cat([prompt_emb, elem], dim = 0)
            list_prompting_elem.append(elem)
        prompting_elem = torch.stack(list_prompting_elem, dim=0)
        if source == 'ViT':
            prompting_elem = prompting_elem.view(-1, prompting_elem.size(2), prompting_elem.size(3))
        return prompting_elem

    def get_task_selector_naive(self, source, x):
        list_prompting_elem = []
        if source == 'ViT':
            x = x.view(x.size(0)//self.num_segments, self.num_segments, x.size(1), x.size(2))
        for i in range(len(self.text_description)):
            cls = self.text_description[i]
            if cls in self.cls_to_task_id:
                task_id = self.cls_to_task_id[cls]
            else:
                cls = cls.replace(' ', '')
                task_id = self.cls_to_task_id[cls]
            L_tk = self.L_sp_tk if source == 'ViT' else self.L_tp_tk
            idx = torch.tensor(list(range(L_tk))).to(self.device)
            prompt_emb = getattr(self, 'P_tk_{}_{}'.format(source, task_id))(idx)
            elem = x[i]
            if source == 'ViT': 
                prompt_emb = prompt_emb.unsqueeze(dim=0)
                prompt_emb = prompt_emb.expand(elem.size(0), prompt_emb.size(1), prompt_emb.size(2))
                elem = torch.cat([prompt_emb, elem], dim = 1)
            else:
                elem = torch.cat([prompt_emb, elem], dim = 0)
            list_prompting_elem.append(elem)
        prompting_elem = torch.stack(list_prompting_elem, dim=0)
        if source == 'ViT':
            prompting_elem = prompting_elem.view(-1, prompting_elem.size(2), prompting_elem.size(3))
        return prompting_elem

    def forward(self, x: torch.Tensor, source = '', type_task = 'TIL'):
        if self.type_prompt == 'general':
            idx = self.idx_P_gn[source]
            P_gn = getattr(self, 'P_gn_{}'.format(source))
            P_gn_sel = P_gn(idx)
            P_gn_sel = P_gn_sel.unsqueeze(dim=0)
            P_gn_sel = P_gn_sel.expand(x.size(0), P_gn_sel.size(1), P_gn_sel.size(2))
            num_seq_org = x.size(1)
            x = torch.cat([P_gn_sel, x], dim = 1)
            if source == 'txt':
                print('text modulation')
                x = x[:, :num_seq_org, :]
        elif self.type_prompt == 'learning_to_prompt':
            print('learning to prompt')
            x = self.get_LP_selector(source, x)
        else:
            if type_task == 'TIL':
                print('TIL selection')
                x = self.get_task_selector_naive(source, x)
            else:
                print('CIL selection')
                x = self.get_task_selector(source, x)
        return x