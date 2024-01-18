# --------------------------------------------------------
# Anchor loss with Wasserstein distance metric along with adaptive anchors motivated from https://github.com/dimitymiller/openset_detection
# This code has been modified from the codes of the above repo.
# Copyright © 2024 Prakash Mallick, AIML
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

def anchor_cross_entropy(distances, 
                    label,
                    weight=None,
                    anchor_weight = 0.1,
                    num_classes = 20,
                    reduction='mean',
                    avg_factor=None,
                    class_weight=None):
    
        """Calculate the CrossEntropy loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, C), C is the number
                of classes.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            class_weight (list[float], optional): The weight for each class.

        Returns:
            torch.Tensor: The calculated loss
        """
        # element-wise losses
        


        if weight is not None:
            weight = weight.float()


        if len(label.size()) > 1:
            label = torch.reshape(label, (-1, num_classes))
            label = torch.argmax(label, dim = 1)
            label = label.long()

        #don't apply to background classes
        #mask = label != num_classes
        # Prakash changed : don't apply to unknown classes and background classes
        array = torch.tensor(np.arange(20,num_classes+1))
        mask = torch.any(torch.eq(label.unsqueeze(0), array.cuda().unsqueeze(1)), dim=0)
        mask=~mask
        if torch.sum(mask) != 0:

            label = label[mask]

            distances = distances[mask]
            #batch_size, num_elements = distances.shape
            #indices = (torch.arange(0, batch_size) * num_elements).cuda() + label
            # Use advanced indexing to gather elements from 'distances'
            #gathered_elements = distances.view(-1)[indices]
            loss_a = torch.gather(distances, 1, label.reshape(-1, 1)).reshape(-1)
            if weight != None:
                weight = weight.reshape(-1)[mask]
                loss_a *= weight

            if reduction == 'mean':

                avg_factor = torch.sum(mask)
                if avg_factor is not None:
                    loss_a = loss_a.sum()/avg_factor
                else:
                    loss_a = torch.mean(loss_a)
            else:
                loss_a = torch.sum(loss_a)
        else:
            loss_a = 0

        return   (anchor_weight*loss_a)

def set_anchors(num_classes,val=5):
        #plus one to account for background class
        
        anch = torch.diag(torch.ones(num_classes+1)*val)
        anch = torch.where(anch != 0, anch, torch.Tensor([-1*val]))
        anchors = nn.Parameter(anch, requires_grad = False).cuda()
        return anchors


class AnchorwCrossEntropyLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0, anchor_weight = 0.1, num_classes = 20):
      
        super(AnchorwCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.num_classes = num_classes
        self.anchor_weight = anchor_weight

        self.anchors = set_anchors(num_classes)
        

    def euclideanDistanceAdaptiveAnc(self, logits):
        #plus one to account for background clss logit

        logits = logits.view(-1, self.num_classes+1)
        #self.anchors = set_anchors(self.num_classes, val= torch.mean(logits).cpu().detach().numpy()  )
        n = logits.size(0)
        m = self.anchors.size(0)
        d = logits.size(1)

        # Define the input measures
        x = logits.detach().requires_grad_(True)  
        y = self.anchors.detach().requires_grad_(True)  

        # Define the Sinkhorn distance
        loss = SamplesLoss("sinkhorn", p=1, blur=0.1, scaling=0.5, backend="tensorized")

        # Compute the Sinkhorn distance between x and y
        dist = loss(x, y)

        # Compute the derivative of the Sinkhorn distance with respect to x
        mean_dist = dist.mean()
        mean_dist.backward(retain_graph=True)
        grad_y = y.grad

        temp = self.anchors - (0.02 * grad_y)
        self.anchors =  nn.Parameter(temp , requires_grad = False).cuda()

        x = logits.unsqueeze(1).expand(n, m, d)
        #anchors = self.anchors.unsqueeze(0).expand(n, m, d)

        dist_wass = dist.unsqueeze(-1).unsqueeze(-1).expand(x.shape[0],self.num_classes,1).squeeze(-1)#.shape
        #for i in range(2048):
        #    a[i]=loss(x[i,:,:],anchors[i,:,:])
        #pdb.set_trace()
        #dist_wass = None
        #dists = torch.norm( x-anchors, 2, 2)
        #del grad_x
        del grad_y
        # Clear the gradient buffers
        x.grad = None
        y.grad = None
        del x,y, temp
        return dist_wass



    def euclideanDistance(self, logits):
        #plus one to account for background clss logit

        logits = logits.view(-1, self.num_classes+1)
        #self.anchors = set_anchors(self.num_classes, val= torch.mean(logits).cpu().detach().numpy()  )
        n = logits.size()[0]
        m = self.anchors.size()[0]
        d = logits.size()[1]

        x = logits.unsqueeze(1).expand(n, m, d)

        anchors = self.anchors.unsqueeze(0).expand(n, m, d)
         
        # Define a Sinkhorn (~Wasserstein) loss between sampled measures
        loss = SamplesLoss("sinkhorn", p=1, blur=0.1, scaling=0.5, backend="tensorized")
        dist_wass = loss(x,anchors)

        dist_wass = dist_wass.unsqueeze(-1).unsqueeze(-1).expand(x.shape[0],self.num_classes,1).squeeze(-1)#.shape
        #for i in range(2048):
        #    a[i]=loss(x[i,:,:],anchors[i,:,:])

        #dist_wass = None
        dists = torch.norm(x-anchors, 2, 2)

        return  dists,dist_wass

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None

        _ , distances = self.euclideanDistance(cls_score)


        # distances = self.euclideanDistance(cls_score[:, :-1])
        #loss_odin = odin_loss(cls_score,label)
        
        loss_cls = self.loss_weight * anchor_cross_entropy(
            distances, 
            label,
            weight = weight,
            num_classes = self.num_classes,
            anchor_weight = self.anchor_weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss_cls
