import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from .map_crop_lr import SingleRoIExtractor_lr
from .map_crop_ud import SingleRoIExtractor_ud


@HEADS.register_module()
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False 
            self.mask_roi_extractor_lr = SingleRoIExtractor_lr()
            self.mask_roi_extractor_ud = SingleRoIExtractor_ud()
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

#    def forward_train(self,
#                      x,
#                      img_metas,
#                      proposal_list,
#                      gt_bboxes,
#                      gt_labels,
#                      gt_bboxes_ignore=None,
#                      gt_masks=None):
#        """
#        Args:
#            x (list[Tensor]): list of multi-level img features.
#            img_metas (list[dict]): list of image info dict where each dict
#                has: 'img_shape', 'scale_factor', 'flip', and may also contain
#                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                For details on the values of these keys see
#                `mmdet/datasets/pipelines/formatting.py:Collect`.
#            proposals (list[Tensors]): list of region proposals.
#            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#            gt_labels (list[Tensor]): class indices corresponding to each box
#            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#                boxes can be ignored when computing the loss.
#            gt_masks (None | Tensor) : true segmentation masks for each box
#                used if the architecture supports a segmentation task.
#
#        Returns:
#            dict[str, Tensor]: a dictionary of loss components
#        """
#         #assign gts and sample proposals
#        if self.with_bbox or self.with_mask:
#            num_imgs = len(img_metas)
#            if gt_bboxes_ignore is None:
#                gt_bboxes_ignore = [None for _ in range(num_imgs)]
#            sampling_results = []
#            for i in range(num_imgs):
#                assign_result = self.bbox_assigner.assign(
#                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
#                    gt_labels[i])
#                sampling_result = self.bbox_sampler.sample(
#                    assign_result,
#                    proposal_list[i],
#                    gt_bboxes[i],
#                    gt_labels[i],
#                    feats=[lvl_feat[i][None] for lvl_feat in x])
#                sampling_results.append(sampling_result)
#
#        losses = dict()
#        # bbox head forward and loss
#        if self.with_bbox:
#            bbox_results = self._bbox_forward_train(x, sampling_results,
#                                                    gt_bboxes, gt_labels,
#                                                    img_metas)
#            losses.update(bbox_results['loss_bbox'])
#
#        # mask head forward and loss
#        if self.with_mask:
#            mask_results = self._mask_forward_train(x, sampling_results,
#                                                    bbox_results['bbox_feats'],
#                                                    gt_masks, img_metas)
#
##            losses.update(mask_results['loss_mask'])
#
#            losses.update(mask_results['loss_mask'])
#            losses.update(mask_results['loss_mask_left'])
#            losses.update(mask_results['loss_mask_right'])
#            losses.update(mask_results['loss_mask_up'])
#            losses.update(mask_results['loss_mask_down'])
##            losses.update(mask_results['loss_mask_add'])
##        print(losses.keys())
#
#
#        return losses

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        
        mask_f = mask_targets.unsqueeze(1)

        
        from .feature_visualization import draw_feature_map
        draw_feature_map(mask_f,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap4/mask/")
#        print(mask_targets.shape)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

#    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
#                            img_metas):
#        """Run forward function and calculate loss for mask head in
#        training."""
#        if not self.share_roi_extractor:
#            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
##            crop_x = 0.5 * (pos_rois[:, 1] + pos_rois[:, 3])
##            pos_rois_crop = pos_rois.clone()
##            pos_rois_crop[:,1] = crop_x
##            print(pos_rois.shape,'          pos_rois')
#            mask_results = self._mask_forward(x, pos_rois)
##            mask_results_crop = self._mask_forward_crop(x,pos_rois_crop)
#        
#        else:
#            pos_inds = []
#            device = bbox_feats.device
#            for res in sampling_results:
#                pos_inds.append(
#                    torch.ones(
#                        res.pos_bboxes.shape[0],
#                        device=device,
#                        dtype=torch.uint8))
#                pos_inds.append(
#                    torch.zeros(
#                        res.neg_bboxes.shape[0],
#                        device=device,
#                        dtype=torch.uint8))
#            pos_inds = torch.cat(pos_inds)
#
#            mask_results = self._mask_forward(
#                x, pos_inds=pos_inds, bbox_feats=bbox_feats)
##            print(mask_results,'             bbbbbbbbbbbbbbbbbbbbbbb')
##            print(mask_all,'             aaaaaaaaaaaaaaaaaaaaa')
#
#
#
#        mask_targets,mask_targets_left,mask_targets_right,mask_targets_up,mask_targets_down = self.mask_head.get_targets(sampling_results, gt_masks,self.train_cfg)
##        from .feature_visualization import draw_feature_map
##        draw_feature_map(mask_targets,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremaps/mask/")
##        print(mask_targets.shape,'   fffffff')
##        mask_targets_add = mask_targets
#
#                                                  
#        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
##        pos_labels_crop = pos_labels.clone()
##        pos_labels = torch.cat((pos_labels,pos_labels_crop),0)
#        
##        loss_mask_add = self.mask_head.loss(mask_results['mask_add'],mask_targets_all, pos_labels)
##        loss_mask_add.update({"loss_mask_add":loss_mask_add.pop("loss_mask")})
#        
#        
#        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
#                                        mask_targets, pos_labels)
#                                        
#
#        loss_mask_left = self.mask_head.loss_left(mask_results['mask_pred_left'],
#                                             mask_targets_left,pos_labels)
#        loss_mask_right = self.mask_head.loss_right(mask_results['mask_pred_right'],
#                                             mask_targets_right,pos_labels)
#        loss_mask_up = self.mask_head.loss_up(mask_results['mask_pred_up'],
#                                             mask_targets_up,pos_labels)
#        loss_mask_down = self.mask_head.loss_down(mask_results['mask_pred_down'],
#                                             mask_targets_down,pos_labels)
##        loss_mask_add = self.mask_head.loss_mask_add(mask_results['mask_pred_add'],
##                                             mask_targets_add,pos_labels)
#        
##        print(loss_all,'         aaaaaaaaaaa')
##        print(loss_mask,'         bbbbbbbbbbbb')
#
#          
#                                               
#
##        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
##        mask_results.update(loss_mask_crop=loss_mask_crop, mask_targets_crop=mask_targets_crop)
##        print(mask_results)
##        return mask_results
##        print(mask_results,'                mask_result')
#        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets, 
#                            loss_mask_left= loss_mask_left, mask_targets_left=mask_targets_left,
#                            loss_mask_right= loss_mask_right, mask_targets_right=mask_targets_right,
#                            loss_mask_up=loss_mask_up, mask_targets_up=mask_targets_up,
#                            loss_mask_down= loss_mask_down, mask_targets_down=mask_targets_down)
##        print(mask_results.keys(),'            crop')
#        return mask_results



#    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
#        """Mask head forward function used in both training and testing."""
#        assert ((rois is not None) ^
#                (pos_inds is not None and bbox_feats is not None))
#        if rois is not None:
#            mask_feats = self.mask_roi_extractor(
#                x[:self.mask_roi_extractor.num_inputs], rois)
#            if self.with_shared_head:
#                mask_feats = self.shared_head(mask_feats)
#
#        else:
#            assert bbox_feats is not None
#            mask_feats = bbox_feats[pos_inds]
##        print(mask_feats,'            aaaaaaaaa')
#        
#        num = mask_feats.shape[0]
#        mask_feats_crop1 = mask_feats[...,:,0:8]
#        mask_feats_crop2 = mask_feats[...,:,5:-1]
##        print(mask_feats_crop1.shape,'       bbbbbbbbbbbbbbbbbbbbbbbbbb')
##        print(mask_feats_crop2.shape,'       cccccccccccccccccccccccccc')
#        x_pad = torch.zeros(num,256,14,6)         
#        x_pad = x_pad.cuda()
#        
#        mask_feats1 = torch.cat((mask_feats_crop1,x_pad),dim=3)
#        mask_feats2 = torch.cat((x_pad,mask_feats_crop1),dim=3)
##        print(mask_feats2.shape,'            dddddddddddddddddddddddddd')
#            
#
#        mask_pred = self.mask_head(mask_feats)
##        mask_pred_crop1 = self.mask_head(mask_feats1)
##        mask_pred_crop2 = self.mask_head(mask_feats2)
#        from .feature_visualization import draw_feature_map
#        draw_feature_map(mask_feats,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap2/mask_feats/")
##        draw_feature_map(mask_feats1,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap2/mask_feats_left/")
##        draw_feature_map(mask_feats2,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap2/mask_feats_right/")
#        draw_feature_map(mask_feats_crop1,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap2/mask_feats_upper/")
#        draw_feature_map(mask_feats_crop2,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap2/mask_feats_bottom/")
##        draw_feature_map(mask_feats_all,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap2/masks/")
#        
#        
##        mask_all = mask_pred + mask_pred_crop1 + mask_pred_crop2
#        
#        
##        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats, mask_pred1=mask_pred_crop1, mask_feats1=mask_feats1,mask_pred2=mask_pred_crop2, mask_feats2=mask_feats)
##        print(mask_results.keys())
#        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
#        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            mask_feats_left,mask_feats_right = self.mask_roi_extractor_lr(x[:self.mask_roi_extractor.num_inputs], rois)
            mask_feats_up,mask_feats_down = self.mask_roi_extractor_ud(x[:self.mask_roi_extractor.num_inputs], rois)
#            mask_feats_all = torch.cat((mask_feats_left,mask_feats_right,mask_feats_up,mask_feats_down),dim=1)

#            draw_feature_map(mask_feats_all,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap1/masks/")
#            print(mask_feats.shape,'       000000')
#            print(mask_feats_left.shape,'       11111111')
#            print(mask_feats_right.shape,'       22222222')
#            print(mask_feats_up.shape,'       33333333')
#            print(mask_feats_down.shape,'       4444444')

            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)

        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]
            

#
        mask_pred = self.mask_head(mask_feats)
        mask_pred_left = self.mask_head(mask_feats_left)
        mask_pred_right = self.mask_head(mask_feats_right)
        mask_pred_up = self.mask_head(mask_feats_up)
        mask_pred_down = self.mask_head(mask_feats_down)
#        print(mask_pred.shape)
#        mask_fusion = mask_pred + mask_pred_left + mask_pred_right + mask_pred_up + mask_pred_down            
#        from .feature_visualization import draw_feature_map
#        draw_feature_map(mask_pred,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap4/mask_feats/")
#        draw_feature_map(mask_pred_left,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap4/mask_feats_left/")
#        draw_feature_map(mask_pred_right,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap4/mask_feats_right/")
#        draw_feature_map(mask_pred_up,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap4/mask_feats_upper/")
#        draw_feature_map(mask_pred_down,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap4/mask_feats_bottom/")
#        draw_feature_map(mask_fusion,save_dir="/home/fanxinyu/mmdetection-master/results_analysis/featuremap4/mask_feats_fusion/")

        

#        mask_pred_add = torch.cat((mask_pred,mask_pred_left,mask_pred_right,mask_pred_up,mask_pred_down),dim=1)

#        print(mask_pred_add.shape,'   add1')
#        print(mask_pred.shape,'   add2')
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats,mask_pred_left=mask_pred_left, mask_feats_left=mask_feats_left,mask_pred_right=mask_pred_right, mask_feats_right=mask_feats_right,mask_pred_up=mask_pred_up,    
        mask_feats_up=mask_feats_up,mask_pred_down=mask_pred_down,mask_feats_down=mask_feats_down)

        return mask_results

#    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
#        """Mask head forward function used in both training and testing."""
#        assert ((rois is not None) ^
#                (pos_inds is not None and bbox_feats is not None))
#        if rois is not None:
#            mask_feats = self.mask_roi_extractor(
#                x[:self.mask_roi_extractor.num_inputs], rois)
#            if self.with_shared_head:
#                mask_feats = self.shared_head(mask_feats)
#        else:
#            assert bbox_feats is not None
#            mask_feats = bbox_feats[pos_inds]
#
#        mask_pred = self.mask_head(mask_feats)
#        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
#        return mask_results

        

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]


################################################################################################################

#import torch
#
#from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
#from ..builder import HEADS, build_head, build_roi_extractor
#from .base_roi_head import BaseRoIHead
#from .test_mixins import BBoxTestMixin, MaskTestMixin
#
#
#@HEADS.register_module()
#class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
#    """Simplest base roi head including one bbox head and one mask head."""
#
#    def init_assigner_sampler(self):
#        """Initialize assigner and sampler."""
#        self.bbox_assigner = None
#        self.bbox_sampler = None
#        if self.train_cfg:
#            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
#            self.bbox_sampler = build_sampler(
#                self.train_cfg.sampler, context=self)
#
#    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
#        """Initialize ``bbox_head``"""
#        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
#        self.bbox_head = build_head(bbox_head)
#
#    def init_mask_head(self, mask_roi_extractor, mask_head):
#        """Initialize ``mask_head``"""
#        if mask_roi_extractor is not None:
#            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
#            self.share_roi_extractor = False
#        else:
#            self.share_roi_extractor = True
#            self.mask_roi_extractor = self.bbox_roi_extractor
#        self.mask_head = build_head(mask_head)
#
#    def forward_dummy(self, x, proposals):
#        """Dummy forward function."""
#        # bbox head
#        outs = ()
#        rois = bbox2roi([proposals])
#        if self.with_bbox:
#            bbox_results = self._bbox_forward(x, rois)
#            outs = outs + (bbox_results['cls_score'],
#                           bbox_results['bbox_pred'])
#        # mask head
#        if self.with_mask:
#            mask_rois = rois[:100]
#            mask_results = self._mask_forward(x, mask_rois)
#            outs = outs + (mask_results['mask_pred'], )
#        return outs
#
#    def forward_train(self,
#                      x,
#                      img_metas,
#                      proposal_list,
#                      gt_bboxes,
#                      gt_labels,
#                      gt_bboxes_ignore=None,
#                      gt_masks=None):
#        """
#        Args:
#            x (list[Tensor]): list of multi-level img features.
#            img_metas (list[dict]): list of image info dict where each dict
#                has: 'img_shape', 'scale_factor', 'flip', and may also contain
#                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                For details on the values of these keys see
#                `mmdet/datasets/pipelines/formatting.py:Collect`.
#            proposals (list[Tensors]): list of region proposals.
#            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#            gt_labels (list[Tensor]): class indices corresponding to each box
#            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#                boxes can be ignored when computing the loss.
#            gt_masks (None | Tensor) : true segmentation masks for each box
#                used if the architecture supports a segmentation task.
#
#        Returns:
#            dict[str, Tensor]: a dictionary of loss components
#        """
#        # assign gts and sample proposals
#        if self.with_bbox or self.with_mask:
#            num_imgs = len(img_metas)
#            if gt_bboxes_ignore is None:
#                gt_bboxes_ignore = [None for _ in range(num_imgs)]
#            sampling_results = []
#            for i in range(num_imgs):
#                assign_result = self.bbox_assigner.assign(
#                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
#                    gt_labels[i])
#                sampling_result = self.bbox_sampler.sample(
#                    assign_result,
#                    proposal_list[i],
#                    gt_bboxes[i],
#                    gt_labels[i],
#                    feats=[lvl_feat[i][None] for lvl_feat in x])
#                sampling_results.append(sampling_result)
#
#        losses = dict()
#        # bbox head forward and loss
#        if self.with_bbox:
#            bbox_results = self._bbox_forward_train(x, sampling_results,
#                                                    gt_bboxes, gt_labels,
#                                                    img_metas)
#            losses.update(bbox_results['loss_bbox'])
#
#        # mask head forward and loss
#        if self.with_mask:
#            mask_results = self._mask_forward_train(x, sampling_results,
#                                                    bbox_results['bbox_feats'],
#                                                    gt_masks, img_metas)
#            losses.update(mask_results['loss_mask'])
#
#        return losses
#
#    def _bbox_forward(self, x, rois):
#        """Box head forward function used in both training and testing."""
#        # TODO: a more flexible way to decide which feature maps to use
#        bbox_feats = self.bbox_roi_extractor(
#            x[:self.bbox_roi_extractor.num_inputs], rois)
#        if self.with_shared_head:
#            bbox_feats = self.shared_head(bbox_feats)
#        cls_score, bbox_pred, mean_norm_logstd = self.bbox_head(bbox_feats)
#
#        bbox_results = dict(
#            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, mean_norm_logstd=mean_norm_logstd )
#        return bbox_results
#
#    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
#                            img_metas):
#        """Run forward function and calculate loss for box head in training."""
#        rois = bbox2roi([res.bboxes for res in sampling_results])
#        bbox_results = self._bbox_forward(x, rois)
#        
#        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
#        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
#        pos_bboxes1 = pos_bboxes_list[0]
#        pos_bboxes2 = pos_bboxes_list[1]
#        pos_bboxes = torch.cat((pos_bboxes1,pos_bboxes2),dim=0)
#        pos_gt_bboxes1 = pos_gt_bboxes_list[0]
#        pos_gt_bboxes2 = pos_gt_bboxes_list[1]
#        pos_gt_bboxes = torch.cat((pos_gt_bboxes1,pos_gt_bboxes2),dim=0)
#
#        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
#                                                  gt_labels, self.train_cfg)
#        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
#                                        bbox_results['bbox_pred'],
#                                        bbox_results['mean_norm_logstd'],rois,
#                                        pos_bboxes,
#                                        pos_gt_bboxes,
#                                        *bbox_targets)
#
#        bbox_results.update(loss_bbox=loss_bbox)
#        return bbox_results
#
#    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
#                            img_metas):
#        """Run forward function and calculate loss for mask head in
#        training."""
#        if not self.share_roi_extractor:
#            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
#            mask_results = self._mask_forward(x, pos_rois)
#        else:
#            pos_inds = []
#            device = bbox_feats.device
#            for res in sampling_results:
#                pos_inds.append(
#                    torch.ones(
#                        res.pos_bboxes.shape[0],
#                        device=device,
#                        dtype=torch.uint8))
#                pos_inds.append(
#                    torch.zeros(
#                        res.neg_bboxes.shape[0],
#                        device=device,
#                        dtype=torch.uint8))
#            pos_inds = torch.cat(pos_inds)
#
#            mask_results = self._mask_forward(
#                x, pos_inds=pos_inds, bbox_feats=bbox_feats)
#
#        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
#                                                  self.train_cfg)
#        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
#        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
#                                        mask_targets, pos_labels)
#
#        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
#        return mask_results
#
#    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
#        """Mask head forward function used in both training and testing."""
#        assert ((rois is not None) ^
#                (pos_inds is not None and bbox_feats is not None))
#        if rois is not None:
#            mask_feats = self.mask_roi_extractor(
#                x[:self.mask_roi_extractor.num_inputs], rois)
#            if self.with_shared_head:
#                mask_feats = self.shared_head(mask_feats)
#        else:
#            assert bbox_feats is not None
#            mask_feats = bbox_feats[pos_inds]
#
#        mask_pred = self.mask_head(mask_feats)
#        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
#        return mask_results
#
#    async def async_simple_test(self,
#                                x,
#                                proposal_list,
#                                img_metas,
#                                proposals=None,
#                                rescale=False):
#        """Async test without augmentation."""
#        assert self.with_bbox, 'Bbox head must be implemented.'
#
#        det_bboxes, det_labels = await self.async_test_bboxes(
#            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
#        bbox_results = bbox2result(det_bboxes, det_labels,
#                                   self.bbox_head.num_classes)
#        if not self.with_mask:
#            return bbox_results
#        else:
#            segm_results = await self.async_test_mask(
#                x,
#                img_metas,
#                det_bboxes,
#                det_labels,
#                rescale=rescale,
#                mask_test_cfg=self.test_cfg.get('mask'))
#            return bbox_results, segm_results
#
#    def simple_test(self,
#                    x,
#                    proposal_list,
#                    img_metas,
#                    proposals=None,
#                    rescale=False):
#        """Test without augmentation."""
#        assert self.with_bbox, 'Bbox head must be implemented.'
#
#        det_bboxes, det_labels = self.simple_test_bboxes(
#            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
#        if torch.onnx.is_in_onnx_export():
#            if self.with_mask:
#                segm_results = self.simple_test_mask(
#                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
#                return det_bboxes, det_labels, segm_results
#            return det_bboxes, det_labels
#
#        bbox_results = [
#            bbox2result(det_bboxes[i], det_labels[i],
#                        self.bbox_head.num_classes)
#            for i in range(len(det_bboxes))
#        ]
#
#        if not self.with_mask:
#            return bbox_results
#        else:
#            segm_results = self.simple_test_mask(
#                x, img_metas, det_bboxes, det_labels, rescale=rescale)
#            return list(zip(bbox_results, segm_results))
#
#    def aug_test(self, x, proposal_list, img_metas, rescale=False):
#        """Test with augmentations.
#
#        If rescale is False, then returned bboxes and masks will fit the scale
#        of imgs[0].
#        """
#        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
#                                                      proposal_list,
#                                                      self.test_cfg)
#
#        if rescale:
#            _det_bboxes = det_bboxes
#        else:
#            _det_bboxes = det_bboxes.clone()
#            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
#                img_metas[0][0]['scale_factor'])
#        bbox_results = bbox2result(_det_bboxes, det_labels,
#                                   self.bbox_head.num_classes)
#
#        # det_bboxes always keep the original scale
#        if self.with_mask:
#            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
#                                              det_labels)
#            return [(bbox_results, segm_results)]
#        else:
#            return [bbox_results]
