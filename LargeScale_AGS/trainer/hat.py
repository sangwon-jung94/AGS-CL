from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td
from PIL import Image
from tqdm import tqdm
import trainer

import networks

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, taskcla):
        super().__init__(model, args, optimizer, evaluator, taskcla)
        
        self.smax=args.lamb
        self.lamb=args.gamma
        self.mask_pre=None
        self.mask_back=None
        
    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

        
    def setup_training(self, lr):
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False

    def train(self, train_loader, test_loader, t):
        
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t>0:
            self.update_frozen_model()
            self.update_fisher()
        
        # Now, you can update self.t
        self.t = t
        kwargs = {'num_workers': 8, 'pin_memory': True}
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False, **kwargs)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True, **kwargs)
        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            idx=0
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.cuda(), target.cuda()
                batch_size = data.shape[0]
                
                task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
                s=(self.smax-1/self.smax)*idx/batch_size+1/self.smax

                output,masks=self.model(data,task,s,mask_return=True)
                loss_CE = self.criterion(output[t],target,masks)

                self.optimizer.zero_grad()
                (loss_CE).backward()
                
                if t>0:
                    for n,p in self.model.named_parameters():
                        if n in self.mask_back:
                            p.grad.data*=self.mask_back[n]
                
                thres_cosh=50
                for n,p in self.model.named_parameters():
                    if n.startswith('e'):
                        num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                        den=torch.cosh(p.data)+1
                        p.grad.data*=self.smax/s*num/den
                
                clipgrad=10000
                torch.nn.utils.clip_grad_norm(self.model.parameters(),clipgrad)
                self.optimizer.step()
                thres_emb=6
                for n,p in self.model.named_parameters():
                    if n.startswith('e'):
                        p.data=torch.clamp(p.data,-thres_emb,thres_emb)
                        
                idx += 1

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            valid_loss,valid_acc=self.evaluator.evaluate(self.model, self.test_iterator, t)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print()
            
        task=torch.autograd.Variable(torch.LongTensor([t]).cuda())
        mask=self.model.mask(task,s=self.smax)
        for i in range(len(mask)):
            mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
        if t==0:
            self.mask_pre=mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

        # Weights mask
        self.mask_back={}
        for n,_ in self.model.named_parameters():
            vals=self.model.get_view_for(n,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals
        
    def criterion(self,outputs,targets,masks):
        reg=0
        count=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        reg/=count
        return self.ce(outputs,targets)+self.lamb*reg
