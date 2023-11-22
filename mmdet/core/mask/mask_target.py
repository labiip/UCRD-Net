import numpy as np
import torch
from torch.nn.modules.utils import _pair


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        list[Tensor]: Mask target of each image.

    Example:
        >>> import mmcv
        >>> import mmdet
        >>> from mmdet.core.mask import BitmapMasks
        >>> from mmdet.core.mask.mask_target import *
        >>> H, W = 17, 18
        >>> cfg = mmcv.Config({'mask_size': (13, 14)})
        >>> rng = np.random.RandomState(0)
        >>> # Positive proposals (tl_x, tl_y, br_x, br_y) for each image
        >>> pos_proposals_list = [
        >>>     torch.Tensor([
        >>>         [ 7.2425,  5.5929, 13.9414, 14.9541],
        >>>         [ 7.3241,  3.6170, 16.3850, 15.3102],
        >>>     ]),
        >>>     torch.Tensor([
        >>>         [ 4.8448, 6.4010, 7.0314, 9.7681],
        >>>         [ 5.9790, 2.6989, 7.4416, 4.8580],
        >>>         [ 0.0000, 0.0000, 0.1398, 9.8232],
        >>>     ]),
        >>> ]
        >>> # Corresponding class index for each proposal for each image
        >>> pos_assigned_gt_inds_list = [
        >>>     torch.LongTensor([7, 0]),
        >>>     torch.LongTensor([5, 4, 1]),
        >>> ]
        >>> # Ground truth mask for each true object for each image
        >>> gt_masks_list = [
        >>>     BitmapMasks(rng.rand(8, H, W), height=H, width=W),
        >>>     BitmapMasks(rng.rand(6, H, W), height=H, width=W),
        >>> ]
        >>> mask_targets = mask_target(
        >>>     pos_proposals_list, pos_assigned_gt_inds_list,
        >>>     gt_masks_list, cfg)
        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = torch.cat(mask_targets)
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    """Compute mask target for each positive proposal in the image.

    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the mask size.

    Returns:
        Tensor: Mask target of each positive proposals in the image.

    Example:
        >>> import mmcv
        >>> import mmdet
        >>> from mmdet.core.mask import BitmapMasks
        >>> from mmdet.core.mask.mask_target import *  # NOQA
        >>> H, W = 32, 32
        >>> cfg = mmcv.Config({'mask_size': (7, 11)})
        >>> rng = np.random.RandomState(0)
        >>> # Masks for each ground truth box (relative to the image)
        >>> gt_masks_data = rng.rand(3, H, W)
        >>> gt_masks = BitmapMasks(gt_masks_data, height=H, width=W)
        >>> # Predicted positive boxes in one image
        >>> pos_proposals = torch.FloatTensor([
        >>>     [ 16.2,   5.5, 19.9, 20.9],
        >>>     [ 17.3,  13.6, 19.3, 19.3],
        >>>     [ 14.8,  16.4, 17.0, 23.7],
        >>>     [  0.0,   0.0, 16.0, 16.0],
        >>>     [  4.0,   0.0, 20.0, 16.0],
        >>> ])
        >>> # For each predicted proposal, its assignment to a gt mask
        >>> pos_assigned_gt_inds = torch.LongTensor([0, 1, 2, 1, 1])
        >>> mask_targets = mask_target_single(
        >>>     pos_proposals, pos_assigned_gt_inds, gt_masks, cfg)
        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
    """
    device = pos_proposals.device
    mask_size = _pair(cfg.mask_size)
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = gt_masks.height, gt_masks.width
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        mask_targets = gt_masks.crop_and_resize(
            proposals_np, mask_size, device=device,
            inds=pos_assigned_gt_inds).to_ndarray()

        mask_targets = torch.from_numpy(mask_targets).float().to(device)
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)

    return mask_targets

#import numpy as np
#import torch
#from torch.nn.modules.utils import _pair
#
#
#def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
#                cfg):
#    """Compute mask target for positive proposals in multiple images.
#
#    Args:
#        pos_proposals_list (list[Tensor]): Positive proposals in multiple
#            images.
#        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
#            positive proposals.
#        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
#            each image.
#        cfg (dict): Config dict that specifies the mask size.
#
#    Returns:
#        list[Tensor]: Mask target of each image.
#
#    Example:
#        >>> import mmcv
#        >>> import mmdet
#        >>> from mmdet.core.mask import BitmapMasks
#        >>> from mmdet.core.mask.mask_target import *
#        >>> H, W = 17, 18
#        >>> cfg = mmcv.Config({'mask_size': (13, 14)})
#        >>> rng = np.random.RandomState(0)
#        >>> # Positive proposals (tl_x, tl_y, br_x, br_y) for each image
#        >>> pos_proposals_list = [
#        >>>     torch.Tensor([
#        >>>         [ 7.2425,  5.5929, 13.9414, 14.9541],
#        >>>         [ 7.3241,  3.6170, 16.3850, 15.3102],
#        >>>     ]),
#        >>>     torch.Tensor([
#        >>>         [ 4.8448, 6.4010, 7.0314, 9.7681],
#        >>>         [ 5.9790, 2.6989, 7.4416, 4.8580],
#        >>>         [ 0.0000, 0.0000, 0.1398, 9.8232],
#        >>>     ]),
#        >>> ]
#        >>> # Corresponding class index for each proposal for each image
#        >>> pos_assigned_gt_inds_list = [
#        >>>     torch.LongTensor([7, 0]),
#        >>>     torch.LongTensor([5, 4, 1]),
#        >>> ]
#        >>> # Ground truth mask for each true object for each image
#        >>> gt_masks_list = [
#        >>>     BitmapMasks(rng.rand(8, H, W), height=H, width=W),
#        >>>     BitmapMasks(rng.rand(6, H, W), height=H, width=W),
#        >>> ]
#        >>> mask_targets = mask_target(
#        >>>     pos_proposals_list, pos_assigned_gt_inds_list,
#        >>>     gt_masks_list, cfg)
#        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
#    """
#    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
#    mask_targets = map(mask_target_single, pos_proposals_list,
#                    pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
#                    
#    mask_targets_left = map(mask_target_single_left, pos_proposals_list,
#                    pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
#                    
#    mask_targets_right = map(mask_target_single_right, pos_proposals_list,
#                    pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
#    
#    mask_targets_up = map(mask_target_single_up, pos_proposals_list,
#                    pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
#                    
#    mask_targets_down = map(mask_target_single_down, pos_proposals_list,
#                    pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
#
#
#    mask_targets = list(mask_targets)
#    mask_targets_left = list(mask_targets_left)
#    mask_targets_right = list(mask_targets_right)
#    mask_targets_up = list(mask_targets_up)
#    mask_targets_down = list(mask_targets_down)
#    
##    print(len(mask_targets0[0]),'            mask_target0000')
##    print(len(mask_targets0[1]),'            mask_target0000')
#    if len(mask_targets) > 0:
#        mask_targets = torch.cat(mask_targets)
#    if len(mask_targets_left) > 0:
#        mask_targets_left = torch.cat(mask_targets_left)
#    if len(mask_targets_right) > 0:
#        mask_targets_right = torch.cat(mask_targets_right)
#    if len(mask_targets_up) > 0:
#        mask_targets_up = torch.cat(mask_targets_up)
#    if len(mask_targets_down) > 0:
#        mask_targets_down = torch.cat(mask_targets_down)
##    print(mask_targets.shape,'         mask_target')
##    print(mask_targets_crop1.shape,'         mask_target1')
#
#    return mask_targets,mask_targets_left,mask_targets_right,mask_targets_up,mask_targets_down
#
#
#def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
#    """Compute mask target for each positive proposal in the image.
#
#    Args:
#        pos_proposals (Tensor): Positive proposals.
#        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
#        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
#            or Polygon.
#        cfg (dict): Config dict that indicate the mask size.
#
#    Returns:
#        Tensor: Mask target of each positive proposals in the image.
#
#    Example:
#        >>> import mmcv
#        >>> import mmdet
#        >>> from mmdet.core.mask import BitmapMasks
#        >>> from mmdet.core.mask.mask_target import *  # NOQA
#        >>> H, W = 32, 32
#        >>> cfg = mmcv.Config({'mask_size': (7, 11)})
#        >>> rng = np.random.RandomState(0)
#        >>> # Masks for each ground truth box (relative to the image)
#        >>> gt_masks_data = rng.rand(3, H, W)
#        >>> gt_masks = BitmapMasks(gt_masks_data, height=H, width=W)
#        >>> # Predicted positive boxes in one image
#        >>> pos_proposals = torch.FloatTensor([
#        >>>     [ 16.2,   5.5, 19.9, 20.9],
#        >>>     [ 17.3,  13.6, 19.3, 19.3],
#        >>>     [ 14.8,  16.4, 17.0, 23.7],
#        >>>     [  0.0,   0.0, 16.0, 16.0],
#        >>>     [  4.0,   0.0, 20.0, 16.0],
#        >>> ])
#        >>> # For each predicted proposal, its assignment to a gt mask
#        >>> pos_assigned_gt_inds = torch.LongTensor([0, 1, 2, 1, 1])
#        >>> mask_targets = mask_target_single(
#        >>>     pos_proposals, pos_assigned_gt_inds, gt_masks, cfg)
#        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
#    """
#    device = pos_proposals.device
#    mask_size = _pair(cfg.mask_size)
#    num_pos = pos_proposals.size(0)
#    if num_pos > 0:
#        proposals_np = pos_proposals.cpu().numpy()
#        maxh, maxw = gt_masks.height, gt_masks.width
#        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
#        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
#        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
#
#        mask_targets = gt_masks.crop_and_resize(
#            proposals_np, mask_size, device=device,
#            inds=pos_assigned_gt_inds).to_ndarray()
#
#        mask_targets = torch.from_numpy(mask_targets).float().to(device)
#    else:
#        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)
#
#    return mask_targets
#    
#def mask_target_single_left(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
#
#    device = pos_proposals.device
#    mask_size = _pair(cfg.mask_size)
#    num_pos = pos_proposals.size(0)
#    if num_pos > 0:
#        proposals_np = pos_proposals.cpu().numpy()
#        scale_x = proposals_np[:,0] + (proposals_np[:,2] - proposals_np[:,0]) * 0.6
##        print(proposals_np.shape,'            proposals_np')
#
#        proposals_np_crop1 = np.copy(proposals_np)
#
#
#        proposals_np_crop1[:,2] = scale_x
#
#
##        proposals_np = np.concatenate((proposals_np,proposals_np_crop1,proposals_np_crop2,proposals_np_crop3,proposals_np_crop4))
##        print(proposals_np_crop1.shape,'            proposals_np_crop1')
#
#        
#        maxh, maxw = gt_masks.height, gt_masks.width
#        proposals_np_crop1[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
#        proposals_np_crop1[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
#        
#
#        
#        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
#
#
#        mask_targets_left = gt_masks.crop_and_resize(
#            proposals_np_crop1, (28,28), device=device,
#            inds=pos_assigned_gt_inds).to_ndarray()
##        print(mask_targets0.shape,'          mask_target00')
#
#        mask_targets_left = torch.from_numpy(mask_targets_left).float().to(device)
##        print(mask_targets0.shape,'            mask_target000')
#    else:
#        mask_targets_left = pos_proposals.new_zeros((0, ) + mask_size)
#
#    return mask_targets_left
#    
#    
#def mask_target_single_right(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
#
#    device = pos_proposals.device
#    mask_size = _pair(cfg.mask_size)
#    # mask_size_crop = _pair(cfg.mask_size_crop)
#    num_pos = pos_proposals.size(0)
#    if num_pos > 0:
#        proposals_np = pos_proposals.cpu().numpy()
#        #        print(proposals_np.shape,'            proposals_np')
#        scale_x = proposals_np[:,0] + (proposals_np[:,2] - proposals_np[:,0]) * 0.6
#        # scale_y = proposals_np[:,1] + (proposals_np[:,1] - proposals_np[:,3]) * 0.64
#        proposals_np_crop2 = np.copy(proposals_np)
#        # proposals_np_crop3 = np.copy(proposals_np)
#        # proposals_np_crop4 = np.copy(proposals_np)
#        proposals_np_crop2[:, 0] = scale_x
#        # proposals_np_crop3[:, 3] = scale_y
#        # proposals_np_crop4[:, 1] = scale_y
#        #        proposals_np = np.concatenate((proposals_np,proposals_np_crop1,proposals_np_crop2,proposals_np_crop3,proposals_np_crop4))
#        #        print(proposals_np_crop1.shape,'            proposals_np_crop1')
#
#        maxh, maxw = gt_masks.height, gt_masks.width
#        proposals_np_crop2[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
#        proposals_np_crop2[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
#
#        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
#
#        mask_targets_right = gt_masks.crop_and_resize(
#            proposals_np_crop2, (28,28), device=device,
#            inds=pos_assigned_gt_inds).to_ndarray()
#        #        print(mask_targets0.shape,'          mask_target00')
#
#        mask_targets_right = torch.from_numpy(mask_targets_right).float().to(device)
#    #        print(mask_targets0.shape,'            mask_target000')
#    else:
#        mask_targets_right = pos_proposals.new_zeros((0,) + mask_size)
#
#    return mask_targets_right
#    
#    
#def mask_target_single_up(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
#    """Compute mask target for each positive proposal in the image.
#
#    Args:
#        pos_proposals (Tensor): Positive proposals.
#        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
#        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
#            or Polygon.
#        cfg (dict): Config dict that indicate the mask size.
#
#    Returns:
#        Tensor: Mask target of each positive proposals in the image.
#
#    Example:
#        >>> import mmcv
#        >>> import mmdet
#        >>> from mmdet.core.mask import BitmapMasks
#        >>> from mmdet.core.mask.mask_target import *  # NOQA
#        >>> H, W = 32, 32
#        >>> cfg = mmcv.Config({'mask_size': (7, 11)})
#        >>> rng = np.random.RandomState(0)
#        >>> # Masks for each ground truth box (relative to the image)
#        >>> gt_masks_data = rng.rand(3, H, W)
#        >>> gt_masks = BitmapMasks(gt_masks_data, height=H, width=W)
#        >>> # Predicted positive boxes in one image
#        >>> pos_proposals = torch.FloatTensor([
#        >>>     [ 16.2,   5.5, 19.9, 20.9],
#        >>>     [ 17.3,  13.6, 19.3, 19.3],
#        >>>     [ 14.8,  16.4, 17.0, 23.7],
#        >>>     [  0.0,   0.0, 16.0, 16.0],
#        >>>     [  4.0,   0.0, 20.0, 16.0],
#        >>> ])
#        >>> # For each predicted proposal, its assignment to a gt mask
#        >>> pos_assigned_gt_inds = torch.LongTensor([0, 1, 2, 1, 1])
#        >>> mask_targets = mask_target_single(
#        >>>     pos_proposals, pos_assigned_gt_inds, gt_masks, cfg)
#        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
#    """
#    device = pos_proposals.device
#    mask_size = _pair(cfg.mask_size)
#    # mask_size_crop = _pair(cfg.mask_size_crop)
#    num_pos = pos_proposals.size(0)
#    if num_pos > 0:
#        proposals_np = pos_proposals.cpu().numpy()
#        #        print(proposals_np.shape,'            proposals_np')
##        scale_x = proposals_np[:,0] + (proposals_np[:,2] - proposals_np[:,0]) * 0.6
#        scale_y = proposals_np[:,1] + (proposals_np[:,1] - proposals_np[:,3]) * 0.6
##        proposals_np_crop2 = np.copy(proposals_np)
#        proposals_np_crop3 = np.copy(proposals_np)
#        # proposals_np_crop4 = np.copy(proposals_np)
##        proposals_np_crop2[:, 0] = scale_x
#        proposals_np_crop3[:, 3] = scale_y
#        # proposals_np_crop4[:, 1] = scale_y
#        #        proposals_np = np.concatenate((proposals_np,proposals_np_crop1,proposals_np_crop2,proposals_np_crop3,proposals_np_crop4))
#        #        print(proposals_np_crop1.shape,'            proposals_np_crop1')
#
#        maxh, maxw = gt_masks.height, gt_masks.width
#        proposals_np_crop3[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
#        proposals_np_crop3[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
#
#        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
#
#        mask_targets_up = gt_masks.crop_and_resize(
#            proposals_np_crop3, (28,28), device=device,
#            inds=pos_assigned_gt_inds).to_ndarray()
#        #        print(mask_targets0.shape,'          mask_target00')
#
#        mask_targets_up = torch.from_numpy(mask_targets_up).float().to(device)
#    #        print(mask_targets0.shape,'            mask_target000')
#    else:
#        mask_targets_up = pos_proposals.new_zeros((0,) + mask_size)
#
#    return mask_targets_up
#    
#def mask_target_single_down(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
#    """Compute mask target for each positive proposal in the image.
#
#    Args:
#        pos_proposals (Tensor): Positive proposals.
#        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
#        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
#            or Polygon.
#        cfg (dict): Config dict that indicate the mask size.
#
#    Returns:
#        Tensor: Mask target of each positive proposals in the image.
#
#    Example:
#        >>> import mmcv
#        >>> import mmdet
#        >>> from mmdet.core.mask import BitmapMasks
#        >>> from mmdet.core.mask.mask_target import *  # NOQA
#        >>> H, W = 32, 32
#        >>> cfg = mmcv.Config({'mask_size': (7, 11)})
#        >>> rng = np.random.RandomState(0)
#        >>> # Masks for each ground truth box (relative to the image)
#        >>> gt_masks_data = rng.rand(3, H, W)
#        >>> gt_masks = BitmapMasks(gt_masks_data, height=H, width=W)
#        >>> # Predicted positive boxes in one image
#        >>> pos_proposals = torch.FloatTensor([
#        >>>     [ 16.2,   5.5, 19.9, 20.9],
#        >>>     [ 17.3,  13.6, 19.3, 19.3],
#        >>>     [ 14.8,  16.4, 17.0, 23.7],
#        >>>     [  0.0,   0.0, 16.0, 16.0],
#        >>>     [  4.0,   0.0, 20.0, 16.0],
#        >>> ])
#        >>> # For each predicted proposal, its assignment to a gt mask
#        >>> pos_assigned_gt_inds = torch.LongTensor([0, 1, 2, 1, 1])
#        >>> mask_targets = mask_target_single(
#        >>>     pos_proposals, pos_assigned_gt_inds, gt_masks, cfg)
#        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
#    """
#    device = pos_proposals.device
#    mask_size = _pair(cfg.mask_size)
#    # mask_size_crop = _pair(cfg.mask_size_crop)
#    num_pos = pos_proposals.size(0)
#    if num_pos > 0:
#        proposals_np = pos_proposals.cpu().numpy()
#        #        print(proposals_np.shape,'            proposals_np')
##        scale_x = proposals_np[:,0] + (proposals_np[:,2] - proposals_np[:,0]) * 0.6
#        scale_y = proposals_np[:,1] + (proposals_np[:,1] - proposals_np[:,3]) * 0.6
##        proposals_np_crop2 = np.copy(proposals_np)
##         proposals_np_crop3 = np.copy(proposals_np)
#        proposals_np_crop4 = np.copy(proposals_np)
##        proposals_np_crop2[:, 0] = scale_x
##         proposals_np_crop3[:, 1] = scale_y
#        proposals_np_crop4[:, 1] = scale_y
#        #        proposals_np = np.concatenate((proposals_np,proposals_np_crop1,proposals_np_crop2,proposals_np_crop3,proposals_np_crop4))
#        #        print(proposals_np_crop1.shape,'            proposals_np_crop1')
#
#        maxh, maxw = gt_masks.height, gt_masks.width
#        proposals_np_crop4[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
#        proposals_np_crop4[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
#
#        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
#
#        mask_targets_down = gt_masks.crop_and_resize(
#            proposals_np_crop4, (28,28), device=device,
#            inds=pos_assigned_gt_inds).to_ndarray()
#        #        print(mask_targets0.shape,'          mask_target00')
#
#        mask_targets_down = torch.from_numpy(mask_targets_down).float().to(device)
#    #        print(mask_targets0.shape,'            mask_target000')
#    else:
#        mask_targets_down = pos_proposals.new_zeros((0,) + mask_size)
#
#    return mask_targets_down