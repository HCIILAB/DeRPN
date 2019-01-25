# --------------------------------------------------------
# Written by Lele Xie, 2018/12/21
# --------------------------------------------------------

import os
import caffe
import yaml
import numpy as np
import numpy.random as npr
import random as rnd
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.bbox_transform import clip_boxes

class DeRPNProposalsLayer(caffe.Layer):

	def setup(self, bottom, top):

		cfg_key = str('TRAIN' if self.phase == 0 else 'TEST') # either 'TRAIN' or 'TEST'
		self.top_N = cfg[cfg_key].DeRPN_top_N
		self.final_top_M = cfg[cfg_key].DeRPN_final_top_M

		layer_params = yaml.load(self.param_str)
		self._feat_stride = layer_params['feat_stride']

		self.do_NMS = True
		self.nms_thresh = 0.7
		self.epsion = 0.00001

		self.height = 0.
		self.width = 0.

		self.start_point = [(self._feat_stride-1)/2.,(self._feat_stride-1)/2.]

		self.anchor_strings = np.array(cfg.DeRPN_anchor_strings,np.float32)
		self.num_anchors = self.anchor_strings.shape[0]

		self.beta = 0.1

		top[0].reshape(1,5)
		# top[1].reshape(1,)


	def forward(self, bottom, top):

		im_info = bottom[2].data[:, :]
		pred_prob = bottom[0].data[:,self.num_anchors*2:,:,:]
		pred_prob += self.epsion
		pred_prob_h = pred_prob[:,:self.num_anchors,:,:]
		pred_prob_w = pred_prob[:,self.num_anchors:,:,:]
		pred_reg = bottom[1].data
		num_images = pred_reg.shape[0]

		yth1, xth1, h_cth1, w_cth1 = self.A_select_B(pred_prob_h, pred_prob_w)
		yth2, xth2, w_cth2, h_cth2 = self.A_select_B(pred_prob_w, pred_prob_h)
		yth = np.hstack((yth1, yth2))
		xth = np.hstack((xth1, xth2))
		h_cth = np.hstack((h_cth1, h_cth2))
		w_cth = np.hstack((w_cth1, w_cth2))
		nth = np.zeros_like(h_cth,np.int32)
		for val in range(h_cth.shape[0]):
			nth[val,:] = val        

		rois, probs = self.eval_rois( yth, xth, h_cth, w_cth, nth, pred_prob, pred_reg)

		all_rois = np.empty((0,5), np.float32)
		all_probs = np.empty((0,), np.float32)

		for im_i in range(num_images):
			inds = np.where(rois[:,0] == im_i)[0]
			cur_proposals = rois[inds, 1:].astype(np.float32)
			cur_probs = probs[inds].astype(np.float32)
			cur_proposals = clip_boxes(cur_proposals, im_info[im_i, :2])

			if self.do_NMS:
				keep = nms(np.hstack((cur_proposals, cur_probs[:,np.newaxis])), np.float(self.nms_thresh) )
				if self.final_top_M > 0:
					keep = keep[:self.final_top_M]
				cur_proposals = cur_proposals[keep, :]
				cur_probs = cur_probs[keep]

			batch_inds = im_i * np.ones(
				(cur_proposals.shape[0], 1), dtype=np.float32
			)
			cur_rois = np.hstack((batch_inds, cur_proposals))
			all_rois = np.concatenate((all_rois, cur_rois), axis=0)
			all_probs = np.concatenate((all_probs, cur_probs), axis=0)

		top[0].reshape(*all_rois.shape)
		top[0].data[...] = all_rois
		# top[1].reshape(*all_probs.shape)
		# top[1].data[...] = all_probs
	def  backward(self, top, propagate_down, bottom):
		pass
		
	def reshape(self, bottom, top):
		pass

	def eval_rois(self, yth, xth, h_cth, w_cth, nth, pred_prob, pred_reg):
		pred_reg_hw = pred_reg[:,:2*self.num_anchors,:,:]
		pred_reg_xy = pred_reg[:,2*self.num_anchors:,:,:]
		pred_reg_h = pred_reg_hw[:,:self.num_anchors,:,:]
		pred_reg_w = pred_reg_hw[:,self.num_anchors:,:,:]
		pred_reg_y = pred_reg_xy[:,:self.num_anchors,:,:]
		pred_reg_x = pred_reg_xy[:,self.num_anchors:,:,:]  

		pred_prob_h = pred_prob[:,:self.num_anchors,:,:]
		pred_prob_w = pred_prob[:,self.num_anchors:,:,:]

		probs = 2./(1./pred_prob_h[nth, h_cth, yth, xth] + 1./pred_prob_w[nth, w_cth, yth, xth])

		ex_center_y = yth*self._feat_stride +self.start_point[1]
		ex_center_x = xth*self._feat_stride +self.start_point[0]
		cy  = pred_reg_y[nth, h_cth,yth,xth] * self.anchor_strings[h_cth] + ex_center_y
		cx  = pred_reg_x[nth, w_cth,yth,xth] * self.anchor_strings[w_cth] + ex_center_x

		hs = np.exp(pred_reg_h[nth, h_cth,yth,xth]) * self.anchor_strings[h_cth]
		ws = np.exp(pred_reg_w[nth, w_cth,yth,xth]) * self.anchor_strings[w_cth]

		x1 = cx - 0.5 * ws
		y1 = cy - 0.5 * hs
		x2 = cx + 0.5 * ws
		y2 = cy + 0.5 * hs

		im_id = nth
		regions = np.concatenate((im_id[np.newaxis,:,:],x1[np.newaxis,:,:],y1[np.newaxis,:,:],x2[np.newaxis,:,:],y2[np.newaxis,:,:], probs[np.newaxis,:,:]), axis=0)
		regions = regions.transpose(1,2,0)
		regions = regions.reshape(-1,6)

		rois = regions[:,:5]
		probs = regions[:,-1]

		return rois, probs

	def A_select_B(self, prob1,prob2):
		num, channel, height, width = prob1.shape

		list_prob1 = prob1.reshape(prob1.shape[0],-1)
		if list_prob1.shape[1]>self.top_N:
			top_N1 = np.argpartition(-list_prob1, self.top_N, axis=1 )[:, :self.top_N] 
		else:
			top_N1 = np.arange(list_prob1.shape[1],dtype=np.int32)
			top_N1 = np.repeat(top_N1[np.newaxis,:], list_prob1.shape[0], axis=0)

		cth1 = top_N1/(height*width)
		temp_rest = top_N1-(cth1*height*width)
		yth = temp_rest/width
		xth = temp_rest-yth*width
		nth = np.zeros_like(cth1,np.int32)
		for val in range(cth1.shape[0]):
			nth[val,:] = val

		cth2 = np.argpartition(-prob2[nth,:,yth,xth], 2, axis=2)
		cth2 = cth2[:,:,:2] 
		cth2 = cth2.transpose(0,2,1).reshape(cth2.shape[0],-1)
		yth = np.hstack((yth, yth))
		xth = np.hstack((xth, xth))
		cth1 = np.hstack((cth1, cth1))

		return yth, xth, cth1, cth2

		
def new_filter_boxes(boxes, min_size, max_size, im_info):
	ws = boxes[:, 2] - boxes[:, 0] + 1
	hs = boxes[:, 3] - boxes[:, 1] + 1
	x_ctr = boxes[:, 0] + ws / 2.
	y_ctr = boxes[:, 1] + hs / 2.
	keep = np.where(
		(ws >= min_size)
		& (hs >= min_size)
		& (x_ctr < im_info[1])
		& (y_ctr < im_info[0])
		& (ws <= max_size)
		& (hs <= max_size)
	)[0]
	return keep