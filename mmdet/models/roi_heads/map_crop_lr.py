import torch
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_map import BaseRoIExtractor


class SingleRoIExtractor_lr(BaseRoIExtractor):
   """Extract RoI features from a single level feature map.

   If there are multiple input feature levels, each RoI is mapped to a level
   according to its scale. The mapping rule is proposed in
   `FPN <https://arxiv.org/abs/1612.03144>`_.

   Args:
       roi_layer (dict): Specify RoI layer type and arguments.
       out_channels (int): Output channels of RoI layers.
       featmap_strides (List[int]): Strides of input feature maps.
       finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
       init_cfg (dict or list[dict], optional): Initialization config dict.
           Default: None
   """

   def __init__(self,
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32],
                finest_scale=56):
       super(SingleRoIExtractor_lr, self).__init__(roi_layer, out_channels,
                                                featmap_strides)
       self.finest_scale = finest_scale
       

   def map_roi_levels(self, rois, num_levels):
       """Map rois to corresponding feature levels by scales.

       - scale < finest_scale * 2: level 0
       - finest_scale * 2 <= scale < finest_scale * 4: level 1
       - finest_scale * 4 <= scale < finest_scale * 8: level 2
       - scale >= finest_scale * 8: level 3

       Args:
           rois (Tensor): Input RoIs, shape (k, 5).
           num_levels (int): Total level number.

       Returns:
           Tensor: Level index (0-based) of each RoI, shape (k, )
       """
       scale = torch.sqrt(
           (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
       target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
       target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
       return target_lvls

   @force_fp32(apply_to=('feats', ), out_fp16=True)
   def forward(self, feats, rois, roi_scale_factor=None):
   
       """Forward function."""

       d_x = rois[:,1] + (rois[:,3]-rois[:,1]) * 0.6
       d_y = rois[:,2] + (rois[:,4]-rois[:,2]) * 0.6
       rois_left = rois.clone()
       rois_left[:,3] = d_x
       rois_right = rois.clone()
       rois_right[:,1] = d_x
#       print(rois,'    aaaaaaaaaaaaa')
#       print(rois_left,'           bbbbbbbbbbbbbbb')
#       print(rois_right,'            cccccccccccc')
#        print(rois_up,'          ddddddddddddddd')
#        print(rois_down,'   eeeeeeeeeeeeeeeee')
#       print(self.roi_layers,'       crop')


       out_size = self.roi_layers[0].output_size
       num_levels = len(feats)
       expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
       if torch.onnx.is_in_onnx_export():
           # Work around to export mask-rcnn to onnx

           roi_feats_left = rois_left[:, :1].clone().detach()
           roi_feats_left = roi_feats_left.expand(*expand_dims)
           roi_feats_left = roi_feats_left.reshape(-1, self.out_channels, *out_size)
           roi_feats_left = roi_feats_left * 0

           roi_feats_right = rois_right[:, :1].clone().detach()
           roi_feats_right = roi_feats_right.expand(*expand_dims)
           roi_feats_right = roi_feats_right.reshape(-1, self.out_channels, *out_size)
           roi_feats_right = roi_feats_right * 0

       else:

           roi_feats_left = feats[0].new_zeros(
               rois_left.size(0), self.out_channels, *out_size)

           roi_feats_right = feats[0].new_zeros(
               rois_right.size(0), self.out_channels, *out_size)

       # TODO: remove this when parrots supports
       if torch.__version__ == 'parrots':

           roi_feats_left.requires_grad = True
           roi_feats_right.requires_grad = True


       if num_levels == 1:
           if len(rois) == 0:
               return roi_feats_left,roi_feats_right
           return self.roi_layers[0](feats[0], rois_left),self.roi_layers[0](feats[0], rois_right)

       target_lvls = self.map_roi_levels(rois, num_levels)

       if roi_scale_factor is not None:
           rois_left = self.roi_rescale(rois_left, roi_scale_factor)
           rois_right = self.roi_rescale(rois_right, roi_scale_factor)


       for i in range(num_levels):
           mask = target_lvls == i
           if torch.onnx.is_in_onnx_export():
               # To keep all roi_align nodes exported to onnx
               # and skip nonzero op
               mask = mask.float().unsqueeze(-1).expand(*expand_dims).reshape(
                   roi_feats.shape)
               roi_feats_t = self.roi_layers[i](feats[i], rois)
               roi_feats_t *= mask
               roi_feats += roi_feats_t
               continue
           inds = mask.nonzero(as_tuple=False).squeeze(1)
           if inds.numel() > 0:
               rois_l = rois_left[inds]
               roi_feats_t_l = self.roi_layers[i](feats[i], rois_l)
               roi_feats_left[inds] = roi_feats_t_l

               rois_r = rois_right[inds]
               roi_feats_t_r = self.roi_layers[i](feats[i], rois_r)
               roi_feats_right[inds] = roi_feats_t_r



           else:
               # Sometimes some pyramid levels will not be used for RoI
               # feature extraction and this will cause an incomplete
               # computation graph in one GPU, which is different from those
               # in other GPUs and will cause a hanging error.
               # Therefore, we add it to ensure each feature pyramid is
               # included in the computation graph to avoid runtime bugs.


               roi_feats_left += sum(
                   x.view(-1)[0]
                   for x in self.parameters()) * 0. + feats[i].sum() * 0.

               roi_feats_right += sum(
                   x.view(-1)[0]
                   for x in self.parameters()) * 0. + feats[i].sum() * 0.
#        print(roi_feats.shape,'           aaaaaaaaaaaaaaaaaaaaa')
#        print(roi_feats_left.shape,'           bbbbbbbbbbbbbbb')
#        print(roi_feats_right.shape,'            cccccccccccc')
#       print(roi_feats_left.shape,'          ddddddddddddddd')
#       print(roi_feats_right.shape,'   eeeeeeeeeeeeeeeee')

#        return roi_feats,roi_feats_left,roi_feats_right,roi_feats_up,roi_feats_down
       return roi_feats_left,roi_feats_right