from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

#from utils import one_hot_embedding
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, num_classes=43):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
    @staticmethod
    def one_hot_embedding(labels, num_classes):
        '''Embedding labels to one-hot form.
        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.
        Returns:
          (tensor) encoded labels, sized [N,#classes].
        '''
        y = torch.eye(num_classes)  # [D,D]
        print (type(labels))
        print (labels)
        return y[labels]            # [N,D]

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = FocalLoss.one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21]
        t = t[:,1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25

        t = FocalLoss.one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        t = t[:,1:]
        t = Variable(t).cuda()

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self,   cls_targets, cls_preds):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = FocalLoss(cls_preds, cls_targets).
        '''
        #batch_size, num_boxes = cls_targets.size()
        N = cls_targets.size(0)
        #num_pos = pos.data.long().sum()

        
        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        #mask = pos_neg.unsqueeze(1).expand_as(cls_preds)
        masked_cls_preds = cls_preds#.view(N,self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/N, cls_loss.data[0]/N), end=' | ')
        loss = (cls_loss)/N
        return loss