# --------------------------------------------------------
# Written by Lele Xie, 2018/12/20
# --------------------------------------------------------
import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from utils.cython_bbox import bbox_overlaps
import random as rnd
from utils.timer import Timer

class GenerateDeRPNLabelsTargetsLayer(caffe.Layer):

    def setup(self,bottom, top):

        layer_params = yaml.load(self.param_str)
        self._feat_stride = layer_params['feat_stride']

        self.fg_thresh = 0.6

        self.height = 0.
        self.width = 0.

        self.least_fg_num = 10
        self.most_fg_num = 30
        self.bg_fg_ratio = 1        
        self.default_bgnum = 20

        self.beta = 0.1   
        self.extend_ratio  = 1.2
        self.epsion = 0.00001
        
        self.min_gt_size = 1
        self.is_suitable_gt = None

        self.start_point = [(self._feat_stride-1)/2.,(self._feat_stride-1)/2.]

        self.anchor_strings = np.array(cfg.DeRPN_anchor_strings,np.float32)
        self.num_anchors = self.anchor_strings.shape[0]

        assert cfg.TRAIN.RPN_NORMALIZE_TARGETS == False, 'Cannot use RPN normalization when using DeRPN !'

        height, width = bottom[0].data.shape[-2:]
        hw_num = self.num_anchors * 2
        top[0].reshape(1,1,hw_num*height,width)
        top[1].reshape(1,2*hw_num,height,width)
        top[2].reshape(1,2*hw_num,height,width)
        top[3].reshape(1,2*hw_num,height,width)
        top[4].reshape(1, 2, hw_num*height,width)

    def forward(self, bottom, top):

        blob_data = []
        for kth in range(len(bottom)):
            blob_data.append(bottom[kth].data)

        num_images = blob_data[0].shape[0]
        assert num_images == 1,"only support one image"

        gt_boxes = blob_data[1]
        gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])
        self.is_suitable_gt = self.determine_gts(gt_boxes)

        height, width = blob_data[0].shape[2:]
        self.height, self.width = height, width

        whole_labels = np.empty((0, self.num_anchors+self.num_anchors, 
                      height, width), dtype = np.int32)
        whole_targets  = np.zeros((0, 2*(self.num_anchors+self.num_anchors),
                      height, width), dtype = np.float32)
        whole_inside_weights = np.zeros((0, 2*(self.num_anchors+self.num_anchors),
                       height, width), dtype = np.float32)
        reg_num_ofchannel = np.zeros((num_images ,self.num_anchors*4,),np.long)
        cls_num_ofchannel = np.zeros((num_images ,self.num_anchors*2,),np.long)

        for ith_im in range(num_images):
            pred_reg = blob_data[3][ith_im,:,:,:]

            labels, targets,  hw_inside_weights, im_i_reg_num, observe_region = \
                                                     self.eval_labels_targets_around_center(gt_boxes)

            labels = self.observe_to_distribute_within_region(observe_region, gt_boxes, pred_reg, labels)
            
            labels, im_i_fg_num_ofchannel, im_i_bg_num_ofchannel = self.sample_fg_bg(labels)

            reg_num_ofchannel[ith_im, :] = im_i_reg_num            
            cls_num_ofchannel[ith_im,:] = im_i_fg_num_ofchannel + im_i_bg_num_ofchannel

            inside_weights = np.concatenate((hw_inside_weights, hw_inside_weights.copy()), axis = 0)
            whole_labels = np.append(whole_labels, labels[np.newaxis, :, :, :], axis=0)
            whole_targets = np.append(whole_targets, targets[np.newaxis, :, :, :], axis=0)
            whole_inside_weights = np.append(whole_inside_weights, inside_weights[np.newaxis, :, :, :], axis=0)

        cls_weights = np.empty(whole_labels.shape, dtype=np.float32)
        reg_weights = np.empty(whole_targets.shape, dtype=np.float32)
        for i in range(cls_num_ofchannel.shape[1]):
            cls_weights[0,i, :, :].fill(cls_num_ofchannel[0,i])
        for i in range(reg_num_ofchannel.shape[1]):
            reg_weights[0,i, :, :].fill(reg_num_ofchannel[0,i])

        reg_weights = reg_weights + self.epsion
        reg_weights = 1. / reg_weights
        whole_outside_weights = reg_weights

        whole_labels = whole_labels.reshape((1, 1, (self.num_anchors+self.num_anchors) * height, width))
        cls_weights = cls_weights.reshape((1, 1, (self.num_anchors+self.num_anchors) * height, width))
        cls_weights = np.concatenate((cls_weights, cls_weights.copy()), axis = 1)

        top[0].reshape(*whole_labels.shape)
        top[0].data[...] = whole_labels
        top[1].reshape(*whole_targets.shape)
        top[1].data[...] = whole_targets
        top[2].reshape(*whole_inside_weights.shape)
        top[2].data[...] = whole_inside_weights        
        top[3].reshape(*whole_outside_weights.shape)
        top[3].data[...] = whole_outside_weights   
        top[4].reshape(*cls_weights.shape)
        top[4].data[...] = cls_weights


    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass


    def match(self, gt_boxes):
            h_ind = []
            w_ind = []
            reg_num_ofchannel = np.zeros((2*self.num_anchors,),np.long)
            cur_gtw = gt_boxes[:,2]-gt_boxes[:,0]
            cur_gth = gt_boxes[:,3]-gt_boxes[:,1]    
            for ith in range(gt_boxes.shape[0]):
                th = self.match_edge_with_anchors(cur_gth[ith],self.anchor_strings)
                tw = self.match_edge_with_anchors(cur_gtw[ith],self.anchor_strings)
                reg_num_ofchannel[th] += 1
                reg_num_ofchannel[tw + self.num_anchors] += 1
                h_ind.append(th)
                w_ind.append(tw)    
            reg_num_ofchannel = np.hstack((reg_num_ofchannel, reg_num_ofchannel))
            return h_ind, w_ind, reg_num_ofchannel

    def match_edge_with_anchors(self,src,anchors):
        if src < anchors[0]:
            return np.array([0,])
        if src >= anchors[-1]:
            return np.array([anchors.shape[0]-1])
        for i in range(anchors.shape[0]):
            if anchors[i]<=src<anchors[i+1]:
                if src<anchors[i]*(2**0.5-self.beta):
                    return np.array([i,])
                if src>anchors[i]*(2**0.5+self.beta):
                    return np.array([i+1,])
                return np.array([i,i+1])

    def eval_obs_region(self, gt_boxes):
            height, width = self.height, self.width
            observe_region = np.zeros((height, width), np.int32)
            scaled_gt = gt_boxes[:,:4].copy()/float(self._feat_stride)
            gt_w = scaled_gt[:, 2] -scaled_gt[:, 0]
            gt_h = scaled_gt[:, 3] -scaled_gt[:, 1]
            gt_cx = (scaled_gt[:, 2] + scaled_gt[:, 0])*0.5
            gt_cy = (scaled_gt[:, 3] + scaled_gt[:, 1])*0.5

            start_x = np.maximum((gt_cx - gt_w*self.extend_ratio*0.5).astype(np.int32), 0)
            end_x = np.minimum((gt_cx + gt_w*self.extend_ratio*0.5).astype(np.int32), width-1)
            start_y = np.maximum((gt_cy - gt_h*self.extend_ratio*0.5).astype(np.int32), 0)
            end_y = np.minimum((gt_cy + gt_h*self.extend_ratio*0.5).astype(np.int32), height-1)
            for ith in range(gt_boxes.shape[0]):                
                observe_region[start_y[ith]:end_y[ith]+1,start_x[ith]:end_x[ith]+1] = 1
            return observe_region, gt_cx, gt_cy

    def eval_labels_targets_around_center(self, src_gt_boxes):
            height, width = self.height, self.width
            labels = np.zeros((2*self.num_anchors, height, width), dtype=np.int32)
            targets = np.zeros(  (4 *self.num_anchors, height, width), dtype=np.float32)
            length_targets = targets[:2*self.num_anchors,:,:]
            center_targets = targets[2*self.num_anchors:,:,:]
            hw_inside_weights = np.zeros((2*self.num_anchors, height, width), dtype=np.float32)

            inds = np.where(self.is_suitable_gt == 1)[0]
            gt_boxes = src_gt_boxes[inds, :]

            h_ind, w_ind, reg_num_ofchannel = self.match(gt_boxes)     

            observe_region, gt_cx_scaled, gt_cy_scaled = self.eval_obs_region(gt_boxes)

            xth = gt_cx_scaled.astype(np.int32)
            yth = gt_cy_scaled.astype(np.int32)
            coord = np.hstack((yth, xth))
            gt_lengths_h = gt_boxes[:,3] - gt_boxes[:,1]
            gt_center_y = (gt_boxes[:,3] + gt_boxes[:,1] )*0.5
            gt_lengths_w = gt_boxes[:,2] - gt_boxes[:,0]
            gt_center_x = (gt_boxes[:,2] + gt_boxes[:,0] )*0.5
            gt_lengths = np.hstack((gt_lengths_h, gt_lengths_w))
            gt_center = np.hstack((gt_center_y, gt_center_x))

            anchor_ind_ar = []         
            gt_ind_ar = []    
            cth = []                

            hw_ind = h_ind+w_ind    
            assert len(hw_ind)  == gt_lengths.shape[0]

            for ith in range(len(hw_ind)):
                if hw_ind[ith].shape[0] == 2:
                    anchor_ind_ar.append(hw_ind[ith][0])
                    anchor_ind_ar.append(hw_ind[ith][1])
                    gt_ind_ar.append(ith)
                    gt_ind_ar.append(ith)
                    if ith <gt_boxes.shape[0]:
                        cth.append(hw_ind[ith][0])
                        cth.append(hw_ind[ith][1])
                    else:
                        cth.append(hw_ind[ith][0] + self.num_anchors) # for w
                        cth.append(hw_ind[ith][1] + self.num_anchors) # for w
                else:
                    anchor_ind_ar.append(hw_ind[ith][0])
                    gt_ind_ar.append(ith)
                    if ith <gt_boxes.shape[0]:
                        cth.append(hw_ind[ith][0])
                    else:
                        cth.append(hw_ind[ith][0] + self.num_anchors) # for w

            anchor_ind_ar = np.array(anchor_ind_ar, np.int32)
            gt_ind_ar = np.array(gt_ind_ar, np.int32)
            cth = np.array(cth, np.int32)

            yth = np.hstack((yth, yth))
            xth = np.hstack((xth, xth))

            assert self.start_point[1]   == self.start_point[0]   
            
            ex_center = coord[gt_ind_ar]*self._feat_stride+self.start_point[1]

            length_targets[cth, yth[gt_ind_ar], xth[gt_ind_ar]] = \
                                        np.log( gt_lengths[gt_ind_ar]/np.float32(self.anchor_strings[anchor_ind_ar]) )
            center_targets[cth, yth[gt_ind_ar], xth[gt_ind_ar]] = \
                                        (gt_center[gt_ind_ar] -ex_center) /np.float32(self.anchor_strings[anchor_ind_ar])

            hw_inside_weights[cth, yth[gt_ind_ar], xth[gt_ind_ar]] = 1
            labels[cth, yth[gt_ind_ar], xth[gt_ind_ar]] = 1

            return labels, targets,  hw_inside_weights, reg_num_ofchannel, observe_region

    def observe_to_distribute_within_region(self, observe_region, src_gt_boxes, pred_reg, labels):
        
        all_y, all_x = np.where(observe_region == 1)   
        pos_num = all_x.shape[0]
        if not pos_num:
            return labels

        t_ind = np.where(self.is_suitable_gt==1)[0]
        gt_boxes = src_gt_boxes[t_ind,:]

        labels_h = labels[:self.num_anchors,:,:]
        labels_w = labels[self.num_anchors:,:,:]
        pred_reg_hw = pred_reg[:(self.num_anchors+self.num_anchors),:,:]
        pred_reg_xy = pred_reg[(self.num_anchors+self.num_anchors):,:,:]
        pred_reg_h = pred_reg_hw[:self.num_anchors,:,:]
        pred_reg_w = pred_reg_hw[self.num_anchors:,:,:]
        pred_reg_y = pred_reg_xy[:self.num_anchors,:,:]
        pred_reg_x = pred_reg_xy[self.num_anchors:,:,:] 

        all_cx = np.empty((0,pos_num), dtype=np.float32)
        all_cy = np.empty((0,pos_num), dtype=np.float32)
        all_h = np.empty((0,pos_num), dtype=np.float32)
        all_w = np.empty((0,pos_num), dtype=np.float32)

        ex_center_y = all_y*self._feat_stride +self.start_point[1]
        ex_center_x = all_x*self._feat_stride +self.start_point[0]

        ref_lengths = np.hstack((self.anchor_strings, self.anchor_strings)).reshape(-1,1)
        all_cy = pred_reg_y[:, all_y, all_x] * ref_lengths[:self.num_anchors,:] + ex_center_y
        all_h = np.exp(pred_reg_h[:,all_y,all_x]) * ref_lengths[:self.num_anchors,:]
        all_cx = pred_reg_x[:, all_y, all_x] * ref_lengths[self.num_anchors:,:] + ex_center_x
        all_w = np.exp(pred_reg_w[:,all_y,all_x]) * ref_lengths[self.num_anchors:,:]

        all_h = np.repeat(all_h, self.num_anchors, axis = 0 )
        all_cy = np.repeat(all_cy, self.num_anchors, axis = 0 )
        all_w = np.repeat(all_w.reshape(1, -1), self.num_anchors, axis = 0).reshape(-1, pos_num)
        all_cx = np.repeat(all_cx.reshape(1, -1), self.num_anchors, axis = 0).reshape(-1, pos_num)

        all_x1 = all_cx - 0.5 * all_w
        all_x2 = all_cx + 0.5 * all_w
        all_y1 = all_cy - 0.5 * all_h
        all_y2 = all_cy + 0.5 * all_h

        all_pred_regions = np.concatenate((\
                all_x1[np.newaxis,:,:],all_y1[np.newaxis,:,:], all_x2[np.newaxis,:,:], all_y2[np.newaxis,:,:]), axis = 0)

        all_pred_regions = all_pred_regions.reshape(4, 1, -1).transpose(2,1,0).reshape(-1,4)

        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_pred_regions,dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float) )
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(all_pred_regions.shape[0]), argmax_overlaps]

        fg_index = np.where(max_overlaps>=self.fg_thresh)[0]

        if not fg_index.shape[0]:
            return labels

        col_index = fg_index / pos_num
        h_index = col_index/self.num_anchors  
        w_index = col_index%self.num_anchors 
        pos_index = fg_index - col_index*pos_num
        y_index = all_y[pos_index]
        x_index = all_x[pos_index]
        labels_h[h_index, y_index, x_index] = 1
        labels_w[w_index, y_index, x_index] = 1

        return labels            

    def  sample_fg_bg(self, labels,):
        
        fg_num_ofchannel = np.zeros((labels.shape[0],),np.long)
        bg_num_ofchannel = np.zeros((labels.shape[0],),np.long)

        for nth in range(labels.shape[0]):
            fg_h, fg_w = np.where(labels[nth,:,:]==1)
            bg_h, bg_w = np.where(labels[nth,:,:]== 0)
            all_fg_num = fg_h.shape[0]
            all_bg_num = bg_h.shape[0]

            # sample fg
            if  all_fg_num > 0:
                effect_ind = np.arange(all_fg_num)
                if all_fg_num>self.most_fg_num :
                    effect_ind = np.array(rnd.sample(effect_ind,self.most_fg_num), np.int32)
                    labels[nth, fg_h, fg_w] = -1
                    labels[nth,fg_h[effect_ind], fg_w[effect_ind]] = 1
                fg_num_ofchannel[nth] = effect_ind.shape[0]

            # sample bg 
            if all_bg_num>0:
                if all_fg_num>=self.least_fg_num :
                    effect_bg_num = min(all_fg_num * self.bg_fg_ratio, all_bg_num)
                elif rnd.random()>0.5 or all_fg_num == 0:
                    effect_bg_num = min(self.default_bgnum, all_bg_num)
                else:
                    effect_bg_num = min(all_fg_num * self.bg_fg_ratio, all_bg_num)

                effect_bg_ind = np.random.randint(low=0, high=all_bg_num, size = effect_bg_num)
                effect_bg_ind = np.array({}.fromkeys(effect_bg_ind).keys(), np.int32)
                labels[nth, bg_h,  bg_w] = -1
                labels[nth, bg_h[effect_bg_ind],  bg_w[effect_bg_ind]] = 0
                bg_num_ofchannel[nth] = effect_bg_ind.shape[0]

        return labels, fg_num_ofchannel, bg_num_ofchannel        

    def  determine_gts(self ,gt_boxes):
        is_suitable_gt = np.zeros((gt_boxes.shape[0]), np.int32)
        gtw = gt_boxes[:,2]-gt_boxes[:,0]
        gth = gt_boxes[:,3]-gt_boxes[:,1]
        keep = np.where((gtw>self.min_gt_size) & (gth>self.min_gt_size))[0]
        is_suitable_gt[keep] = 1
        return is_suitable_gt