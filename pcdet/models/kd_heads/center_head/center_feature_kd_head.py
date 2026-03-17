import torch
from pcdet.models.kd_heads.kd_head import KDHeadTemplate

from pcdet.models.model_utils.rotated_roi_grid_pool import RotatedGridPool
from pcdet.utils.kd_utils import kd_utils
from pcdet.utils import common_utils, loss_utils
from torchvision.ops import nms


class CenterFeatureKDHead(KDHeadTemplate):
    def __init__(self, model_cfg, dense_head):
        super().__init__(model_cfg, dense_head)
        if self.model_cfg.get('FEATURE_KD'):
            self._init_feature_kd_head(dense_head)
        
        # self.feature_dist = common_utils.AverageMeter()
        # self.feature_dist_top100 = common_utils.AverageMeter()
        # self.feature_dist_spatial = common_utils.AverageMeter()

    def _init_feature_kd_head(self, dense_head):
        if self.model_cfg.FEATURE_KD.get('ROI_POOL', None) and self.model_cfg.FEATURE_KD.ROI_POOL.ENABLED:
            self.roi_pool_func = RotatedGridPool(
                dense_head.point_cloud_range, self.model_cfg.FEATURE_KD.ROI_POOL
            )

    def multi_teacher_roi_pool(self, feature_teas, rois, voxel_size_tea, feature_map_stride_tea):  # features
        """
        Apply rotated grid pooling on features from multiple teachers.

        Args:
            feature_teas: List of tensors [(B, C1, H1, W1), (B, C2, H2, W2), ...]
                          A list of feature maps from multiple teachers.
            rois: (B, num_rois, 7 + C) or a list [num_rois, 7 + C]
                  Region of interests.
            voxel_size_tea: Voxel size for the teacher model(s).
            feature_map_stride_tea: Stride of the feature map for the teacher model(s).

        Returns:
            Tensor of pooled features aggregated from all teachers.
        """
        pooled_features_list = []

        # Iterate over each teacher's feature map and apply the pooling function
        for feature_tea in feature_teas:
            roi_feats_tea = self.roi_pool_func(feature_tea, rois, voxel_size_tea, feature_map_stride_tea)
            pooled_features_list.append(roi_feats_tea)

        # Aggregate the pooled features from all teachers
        # Here we simply concatenate along the channel dimension assuming they have the same spatial dimensions
        roi_feats_agg = torch.cat(pooled_features_list, dim=1)

        return roi_feats_agg

    @staticmethod
    def calculate_feature_rois_aligned(kd_fg_mask, corners_3d):
        """
        Given corner points in 3D, filling the kd fg mask

        Args:
            kd_fg_mask: [h, w]
            corners_3d: [4, 2]. [num_boxes, corners in bev, x,y], position of corner points in BEV coordinates

        Returns:

        """
        left = corners_3d[:, 0].min().floor().int()
        right = corners_3d[:, 0].max().ceil().int()

        top = corners_3d[:, 1].min().floor().int()
        bottom = corners_3d[:, 1].max().ceil().int()

        kd_fg_mask[top:bottom, left:right] = 1

    def build_feature_kd_loss(self):
        if self.model_cfg.KD_LOSS.FEATURE_LOSS.type in ['SmoothL1Loss', 'MSELoss', 'KLDivLoss']:
            self.kd_feature_loss_func = getattr(torch.nn, self.model_cfg.KD_LOSS.FEATURE_LOSS.type)(reduction='none')
        elif self.model_cfg.KD_LOSS.FEATURE_LOSS.type in ['CosineLoss']:
            self.kd_feature_loss_func = getattr(loss_utils, self.model_cfg.KD_LOSS.FEATURE_LOSS.type)()
        else:
            raise NotImplementedError

    def get_feature_kd_loss(self, batch_dict, tb_dict, loss_cfg):
        if loss_cfg.mode == 'raw':
            kd_feature_loss = self.get_feature_kd_loss_raw(batch_dict, loss_cfg)
        elif loss_cfg.mode == 'rois':
            if loss_cfg.tea_num == 'tea_5' and (batch_dict['temperature'] != 0.0):
                kd_feature_loss = self.get_feature_kd_loss_rois3(batch_dict, loss_cfg)
            # ------------------------------------TEA_2--------------------------------------------------------
            elif loss_cfg.tea_num == 'tea_2':
                # feature_name_tea = self.model_cfg.FEATURE_KD.get('FEATURE_NAME_TEA', self.model_cfg.FEATURE_KD.FEATURE_NAME)
                # if batch_dict.get(feature_name_tea + '_tea2', None):
                kd_feature_loss = self.get_feature_kd_loss_rois(batch_dict, loss_cfg)+self.get_feature_kd_loss_rois2(batch_dict, loss_cfg)  # sum
            else:
                kd_feature_loss = self.get_feature_kd_loss_rois(batch_dict, loss_cfg)

        elif loss_cfg.mode == 'spatial':
            kd_feature_loss = self.get_feature_kd_loss_spatial(batch_dict, loss_cfg)
        elif loss_cfg.mode == 'aff':
            kd_feature_loss = self.get_feature_kd_loss_affinity(batch_dict, loss_cfg)
        else:
            raise NotImplementedError

        tb_dict['kd_feat_ls'] = kd_feature_loss if isinstance(kd_feature_loss, float) else kd_feature_loss.item()

        return kd_feature_loss, tb_dict

    def get_feature_kd_loss_raw(self, batch_dict, loss_cfg):
        """
        Args:
            batch_dict:
            loss_cfg
        Returns:

        """
        feature_name = self.model_cfg.FEATURE_KD.FEATURE_NAME
        feature_stu = batch_dict[feature_name]
        feature_name_tea = self.model_cfg.FEATURE_KD.get('FEATURE_NAME_TEA', feature_name)
        feature_tea = batch_dict[feature_name_tea + '_tea']

        target_dicts = batch_dict['target_dicts_tea']

        if feature_stu.shape != feature_tea.shape and self.model_cfg.FEATURE_KD.get('ALIGN', None):
            feature_tea, feature_stu = self.align_feature_map(
                feature_tea, feature_stu, align_cfg=self.model_cfg.FEATURE_KD.ALIGN
            )
        # ----- =align_channel ----------
        feature_tea = feature_tea.transpose(3, 1)
        # box_pred_stu = box_pred_stu.transpose(3, 1)
        f_s = (feature_stu.size(2), feature_stu.size(1))  # h,c
        feature_tea = torch.nn.functional.interpolate(feature_tea, size=f_s, mode='bilinear')
        feature_tea = feature_tea.transpose(3, 1).contiguous()
        assert feature_tea.shape == feature_stu.shape

        # whole feature map mimicking
        bs, channel, height, width = feature_tea.shape
        feature_mask = torch.ones([bs, height, width], dtype=torch.float32).cuda()
        if loss_cfg.get('fg_mask', None):
            fg_mask = self.cal_fg_mask_from_target_heatmap_batch(
                target_dict=target_dicts, soft=loss_cfg.get('soft_mask', None)
            )[0]
            feature_mask *= fg_mask

        if loss_cfg.get('score_mask', None):
            score_mask = self.cal_score_mask_from_teacher_pred(batch_dict['pred_tea'], loss_cfg.score_thresh)[0]
            feature_mask *= score_mask

        kd_feature_loss_all = self.kd_feature_loss_func(feature_stu, feature_tea)
        kd_feature_loss = (kd_feature_loss_all * feature_mask.unsqueeze(1)).sum() / (feature_mask.sum() * channel + 1e-6)

        kd_feature_loss = kd_feature_loss * loss_cfg.weight

        return kd_feature_loss

    def get_feature_kd_loss_rois(self, batch_dict, loss_cfg):
        feature_name = self.model_cfg.FEATURE_KD.FEATURE_NAME
        feature_stu = batch_dict[feature_name]
        feature_name_tea = self.model_cfg.FEATURE_KD.get('FEATURE_NAME_TEA', feature_name)
        feature_tea = batch_dict[feature_name_tea + '_tea']

        feat_height = feature_stu.shape[2]
        feat_height_tea = feature_tea.shape[2]

        bs = feature_stu.shape[0]  # batchsize
        if self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'gt':
            rois = batch_dict['gt_boxes'].detach()
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'tea':
            rois = []
            weis = []
            for b_idx in range(bs):
                cur_pred_tea = batch_dict['decoded_pred_tea'][b_idx]
                pred_scores = cur_pred_tea['pred_scores']
                score_mask = pred_scores > self.model_cfg.FEATURE_KD.ROI_POOL.THRESH
                rois.append(cur_pred_tea['pred_boxes'][score_mask])
                weis.append(pred_scores[score_mask])
                # import pdb; pdb.set_trace()
                # weis.append(pred_scores[score_mask] + 1 / (pred_scores[score_mask] + 10))
            weis = torch.cat(weis)
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'tea_5':
            rois = []
            for b_idx in range(bs):
                cur_pred_tea = batch_dict['decoded_pred_tea'][b_idx]

                # Filter predictions by score threshold
                score_mask = cur_pred_tea['pred_scores'] > self.model_cfg.FEATURE_KD.ROI_POOL.THRESH
                all_scores = torch.tensor(cur_pred_tea['pred_scores'][score_mask], dtype=torch.float32)
                all_boxes = torch.tensor(cur_pred_tea['pred_boxes'][score_mask], dtype=torch.float32)

                if len(all_boxes) > 0:
                    from ....ops.iou3d_nms import iou3d_nms_utils

                    # Compute IoU matrix
                    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(all_boxes[:, :7], all_boxes[:, :7])
                    nms_thresh = self.model_cfg.FEATURE_KD.ROI_POOL.get('NMS_THRESH', 0.1)

                    # Sort boxes by score
                    scores_sorted, indices = torch.sort(all_scores, descending=True)

                    # NMS process
                    keep_indices = []
                    while indices.numel() > 0:
                        keep_indices.append(indices[0])
                        if indices.numel() == 1:
                            break

                        # Calculate IoU with other boxes
                        cur_ious = iou_matrix[indices[0]][indices[1:]]

                        # Keep boxes with low IoU
                        mask = cur_ious < nms_thresh
                        indices = indices[1:][mask]

                    # Get final boxes
                    keep_indices = torch.stack(keep_indices)
                    nms_boxes = all_boxes[keep_indices]

                    # Limit max ROIs
                    max_rois = self.model_cfg.FEATURE_KD.ROI_POOL.get('MAX_ROIS', 100)
                    nms_boxes = nms_boxes[:max_rois]

                    rois.append(nms_boxes)
                else:
                    rois.append(torch.zeros((0, 7)).to(feature_stu_device))
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'stu':
            rois = []
            weis = []
            for b_idx in range(bs):
                cur_pred_tea = batch_dict['decoded_pred_tea'][b_idx]
                pred_scores = cur_pred_tea['pred_scores']
                score_mask = pred_scores > self.model_cfg.FEATURE_KD.ROI_POOL.THRESH
                rois.append(cur_pred_tea['pred_boxes'][score_mask])
                weis.append(pred_scores[score_mask]+1/(pred_scores[score_mask]+2))
            weis = torch.cat(weis)
            pred_dict_stu = self.dense_head.forward_ret_dict['decoded_pred_dicts']
            rois_stu = [pred_dict_stu[i]['pred_boxes'] for i in range(bs)]
            # import pdb; pdb.set_trace()
            weis_stu = [pred_dict_stu[i]['pred_boxes'] for i in range(bs)]
        else:
            raise NotImplementedError

        if feature_stu.shape[2] == feat_height_tea:
            voxel_size_stu = self.voxel_size_tea
            feature_map_stride_stu = self.feature_map_stride_tea
        elif feature_stu.shape[2] == feat_height:
            voxel_size_stu = self.voxel_size
            feature_map_stride_stu = self.feature_map_stride
        else:
            raise NotImplementedError

        if feature_tea.shape[2] == feat_height_tea:
            voxel_size_tea = self.voxel_size_tea
            feature_map_stride_tea = self.feature_map_stride_tea
        elif feature_tea.shape[2] == feat_height:
            voxel_size_tea = self.voxel_size
            feature_map_stride_tea = self.feature_map_stride
        else:
            raise NotImplementedError

        num_rois = 0
        for roi in rois:
            num_rois += roi.shape[0]

        if num_rois == 0:
            kd_feature_loss = 0.0
        else:
            roi_feats = self.roi_pool_func(
                feature_stu, rois, voxel_size_stu, feature_map_stride_stu
            )
            roi_feats_tea = self.roi_pool_func(
                feature_tea, rois, voxel_size_tea, feature_map_stride_tea
            )

            # # # -----  align_channel ----------
            # roi_feats_tea = roi_feats_tea.transpose(3, 1)
            # f_s = (roi_feats.size(2), roi_feats.size(1))  # h,c
            # roi_feats_tea = torch.nn.functional.interpolate(roi_feats_tea, size=f_s, mode='bilinear')
            # roi_feats_tea = roi_feats_tea.transpose(3, 1)
            # kd_feature_loss = loss_cfg.weight * self.kd_feature_loss_func(roi_feats, roi_feats_tea).mean()
            kd_feature_loss=0
            if loss_cfg.get('GID', None):
                cnt = 0
                kd_feat_rel_loss = 0
                for b_roi in rois:
                    num_roi = (b_roi[:, 3] != 0).sum()
                    cur_roi_feats = roi_feats[cnt:cnt + num_roi].view(num_roi, -1)
                    cur_roi_feats_tea = roi_feats_tea[cnt:cnt+num_roi].view(num_roi, -1)

                    rel_tea = common_utils.pair_distance_gpu(cur_roi_feats_tea, cur_roi_feats_tea)
                    rel_tea /= rel_tea.mean()
                    rel_stu = common_utils.pair_distance_gpu(cur_roi_feats, cur_roi_feats)
                    rel_stu /= rel_stu.mean()

                    kd_feat_rel_loss += torch.nn.functional.smooth_l1_loss(rel_tea, rel_stu)
                    cnt += num_roi

                kd_feature_loss += loss_cfg.GID.rel_weight * kd_feat_rel_loss / bs
            if loss_cfg.get('GID_ANG', None):
                cnt = 0
                kd_feat_ang_loss = 0
                #import pdb; pdb.set_trace()
                # -----------------------------------------------------------------------------
                for b_roi in rois:
                    num_roi = (b_roi[:, 3] != 0).sum()
                    cur_roi_feats = roi_feats[cnt:cnt + num_roi].contiguous().view(num_roi, -1)
                    cur_roi_feats_tea = roi_feats_tea[cnt:cnt + num_roi].contiguous().view(num_roi, -1)

                    ang_tea = common_utils.pair_angle_gpu(cur_roi_feats_tea)
                    ang_stu = common_utils.pair_angle_gpu(cur_roi_feats)

                    if self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'tea':
                        loss_ang = torch.nn.functional.smooth_l1_loss(ang_tea, ang_stu, reduction='none')
                        loss_ang = loss_ang.sum(dim=1)
                        kd_feat_ang_loss += torch.dot(loss_ang, weis[cnt:cnt + num_roi])/num_roi
                    else:
                        kd_feat_ang_loss += torch.nn.functional.smooth_l1_loss(ang_tea, ang_stu)
                    cnt += num_roi
                kd_feature_loss += loss_cfg.ang_weight * kd_feat_ang_loss /bs

        return kd_feature_loss

    def get_feature_kd_loss_rois2(self, batch_dict, loss_cfg):
        feature_name = self.model_cfg.FEATURE_KD.FEATURE_NAME  # spatial_features_2d
        feature_stu = batch_dict[feature_name]
        feature_name_tea = self.model_cfg.FEATURE_KD.get('FEATURE_NAME_TEA', feature_name)  # spatial_features_2d
        # --------TEA_2--------------
        # if loss_cfg.get('tea_num', None):
        #     feature_tea = batch_dict[feature_name_tea + '_tea2']  # spatial_features_2d_tea2
        feature_tea = batch_dict[feature_name_tea + '_tea2']

        feat_height = feature_stu.shape[2]
        feat_height_tea = feature_tea.shape[2]

        bs = feature_stu.shape[0]
        if self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'gt':
            rois = batch_dict['gt_boxes'].detach()
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'tea':
            rois = []
            # --------TEA_2--------------
            if loss_cfg.get('tea_new', None):
                for b_idx in range(bs):
                    cur_pred_tea = batch_dict['decoded_pred_tea2'][b_idx]
                    pred_scores = cur_pred_tea['pred_scores']
                    score_mask = pred_scores > self.model_cfg.FEATURE_KD.ROI_POOL.THRESH
                    rois.append(cur_pred_tea['pred_boxes'][score_mask])
            else:
                for b_idx in range(bs):
                    cur_pred_tea = batch_dict['decoded_pred_tea'][b_idx]
                    pred_scores = cur_pred_tea['pred_scores']
                    score_mask = pred_scores > self.model_cfg.FEATURE_KD.ROI_POOL.THRESH
                    rois.append(cur_pred_tea['pred_boxes'][score_mask])
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'stu':
            pred_dict_stu = self.dense_head.forward_ret_dict['decoded_pred_dicts']
            rois = [pred_dict_stu[i]['pred_boxes'] for i in range(bs)]
        else:
            raise NotImplementedError

        if feature_stu.shape[2] == feat_height_tea:
            voxel_size_stu = self.voxel_size_tea
            feature_map_stride_stu = self.feature_map_stride_tea
        elif feature_stu.shape[2] == feat_height:
            voxel_size_stu = self.voxel_size
            feature_map_stride_stu = self.feature_map_stride
        else:
            raise NotImplementedError

        if feature_tea.shape[2] == feat_height_tea:
            voxel_size_tea = self.voxel_size_tea
            feature_map_stride_tea = self.feature_map_stride_tea
        elif feature_tea.shape[2] == feat_height:
            voxel_size_tea = self.voxel_size
            feature_map_stride_tea = self.feature_map_stride
        else:
            raise NotImplementedError

        num_rois = 0
        for roi in rois:
            num_rois += roi.shape[0]
        if num_rois == 0:
            kd_feature_loss = 0.0
        else:
            roi_feats = self.roi_pool_func(
                feature_stu, rois, voxel_size_stu, feature_map_stride_stu
            )
            roi_feats_tea = self.roi_pool_func(
                feature_tea, rois, voxel_size_tea, feature_map_stride_tea
            )
            # ============alian channel xiugai ==================================
            roi_feats = roi_feats.transpose(3, 1)
            f_s = (roi_feats_tea.size(2), roi_feats_tea.size(1))
            roi_feats = torch.nn.functional.interpolate(roi_feats, size=f_s, mode='bilinear')
            roi_feats = roi_feats.transpose(3, 1)
            # ----------------------------------------------------------------
            kd_feature_loss = loss_cfg.weight * self.kd_feature_loss_func(roi_feats, roi_feats_tea).mean()

            if loss_cfg.get('GID', None):
                cnt = 0
                kd_feat_rel_loss = 0
                for b_roi in rois:
                    num_roi = (b_roi[:, 3] != 0).sum()
                    cur_roi_feats = roi_feats[cnt:cnt + num_roi].view(num_roi, -1)
                    cur_roi_feats_tea = roi_feats_tea[cnt:cnt + num_roi].view(num_roi, -1)

                    rel_tea = common_utils.pair_distance_gpu(cur_roi_feats_tea, cur_roi_feats_tea)
                    rel_tea /= rel_tea.mean()
                    rel_stu = common_utils.pair_distance_gpu(cur_roi_feats, cur_roi_feats)
                    rel_stu /= rel_stu.mean()

                    kd_feat_rel_loss += torch.nn.functional.smooth_l1_loss(rel_tea, rel_stu)
                    cnt += num_roi

                kd_feature_loss += loss_cfg.GID.rel_weight * kd_feat_rel_loss / bs

            if loss_cfg.get('GID_ANG', None):
                kd_feature_loss = 0.0
                cnt = 0
                kd_feat_ang_loss = 0
                #import pdb; pdb.set_trace()
                # -----------------------------------------------------------------------------
                for b_roi in rois:
                    num_roi = (b_roi[:, 3] != 0).sum()
                    if num_roi == 0:
                        continue
                    cur_roi_feats = roi_feats[cnt:cnt + num_roi].contiguous().view(num_roi, -1)
                    cur_roi_feats_tea = roi_feats_tea[cnt:cnt + num_roi].view(num_roi, -1)

                    ang_tea = common_utils.pair_angle_gpu(cur_roi_feats_tea)
                    ang_stu = common_utils.pair_angle_gpu(cur_roi_feats)


                    if self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'tea':
                        loss_ang = torch.nn.functional.smooth_l1_loss(ang_tea, ang_stu, reduction='none').sum(dim=1)
                        kd_feat_ang_loss += torch.dot(loss_ang, weis[cnt:cnt + num_roi]) / num_roi
                    else:
                        kd_feat_ang_loss += torch.nn.functional.smooth_l1_loss(ang_tea, ang_stu)
                    cnt += num_roi
                kd_feature_loss += loss_cfg.ang_weight * kd_feat_ang_loss / bs

        return kd_feature_loss

    def get_normalized_weights(self, adaptive_weights, temperature=2.0, adaptive_mask=None):
        tensor_weights = torch.tensor(adaptive_weights, dtype=torch.float32)
        if adaptive_mask is None:
            adaptive_mask= torch.ones_like(tensor_weights)
        tensor_weights = tensor_weights*adaptive_mask
        softmax_weights = torch.nn.functional.softmax(-tensor_weights / temperature, dim=0)  #行方向(dim=0)
        return softmax_weights
    def get_confidence_weights(self, adaptive_weights, temperature=2.0, adaptive_mask=None):
        tensor_weights = torch.tensor(adaptive_weights, dtype=torch.float32)
        if adaptive_mask is None:
            adaptive_mask = torch.ones_like(tensor_weights)
        tensor_weights = tensor_weights * adaptive_mask
        sum_weights = torch.exp(tensor_weights).sum()
        softmax_weights = (1 - torch.exp(tensor_weights) / sum_weights) / tensor_weights.shape[0]
        return softmax_weights
    def create_softmax_mask(self, adaptive_weights, max_thred=5.0, min_thred=0.1):
        tensor_weights = torch.tensor(adaptive_weights, dtype=torch.float32)
        max_val = torch.max(tensor_weights)
        min_val = torch.min(tensor_weights)
        if min_val <= min_thred:
            # Create mask where values < 0.1 are 1, others are 0
            mask = (tensor_weights <= min_thred).float()
        if max_val >= max_thred:
            # Create mask where values > 10 are 0, others are 1
            mask = (tensor_weights <= max_thred).float()
        else:
            # If neither condition is met, return all ones
            mask = torch.ones_like(tensor_weights)
        return mask
        
    def _apply_teacher_alignment(self, features_tea):
        if hasattr(self.dense_head, 'teacher_align_layer'):
            aligned = []
            for feat, align_layer in zip(features_tea, self.dense_head.teacher_align_layer):
                feat256, feat32 = align_layer(feat)
                aligned.append((feat256, feat32))
            return aligned  # len=5, each=(256,32)
    def _apply_student_multi_scale_alignment(self, feature_stu):
        if hasattr(self.dense_head, 'student_align_layer'):
            feat256, feat32 = self.dense_head.student_align_layer(feature_stu)
            return [(feat256, feat32)] * 5  # student 是共享的，不是 5 套独立参数

    def get_feature_kd_loss_rois5(self, batch_dict, loss_cfg):
        feature_name = self.model_cfg.FEATURE_KD.FEATURE_NAME  # spatial_features_2d
        feature_stu = batch_dict[feature_name]
        feature_name_tea = self.model_cfg.FEATURE_KD.get('FEATURE_NAME_TEA', feature_name)  # spatial_features_2d
        features_tea = [
            batch_dict[feature_name_tea + '_tea'],  # teacher 1
            batch_dict[feature_name_tea + '_tea2'],  # teacher 2
            batch_dict[feature_name_tea + '_tea3'],  # teacher 3
            batch_dict[feature_name_tea + '_tea4'],  # teacher 4
            batch_dict[feature_name_tea + '_tea5']  # teacher 5
        ] 
        feat_height = feature_stu.shape[2]
        feat_height_tea = features_tea[0].shape[2] 
        bs = feature_stu.shape[0]
        if self.model_cfg.FEATURE_KD.get('Channel_Align_Layer', None):
            # 检查是否有对齐层
            has_teacher_align = hasattr(self.dense_head, 'teacher_align_layer')
            has_student_align = hasattr(self.dense_head, 'student_align_layer')
            if has_teacher_align and has_student_align:
                features_tea = self._apply_teacher_alignment(features_tea)
                features_stu = self._apply_student_multi_scale_alignment(feature_stu)
            else:
                print(f"[Warning] Align layers not fully registered: "
                      f"teacher={has_teacher_align}, student={has_student_align}")
                return {
                    'feat_kd_loss_total': torch.tensor(0.0, device=feature_stu.device,
                                                       requires_grad=True)
                }

        # ROI selection
        if self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'gt':
            rois = batch_dict['gt_boxes'].detach()
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'tea_5' \
                and (batch_dict['temperature'] == 200 or batch_dict['temperature'] == 0.02):
            rois = []
            for b_idx in range(bs):
                # 从5个teacher中选择最好的预测框
                pred_boxes_list = []
                pred_scores_list = []
                for teacher_idx in range(5):
                    cur_pred_tea = batch_dict[f'decoded_pred_tea{teacher_idx + 1 if teacher_idx > 0 else ""}'][b_idx]
                    pred_scores = cur_pred_tea['pred_scores']
                    score_mask = pred_scores > self.model_cfg.FEATURE_KD.ROI_POOL.THRESH
                    pred_boxes_list.append(cur_pred_tea['pred_boxes'][score_mask])
                    pred_scores_list.append(pred_scores[score_mask])

                # 合并所有teacher的预测框
                if len(pred_boxes_list) > 0:
                    all_boxes = torch.cat(pred_boxes_list, dim=0)
                    all_scores = torch.cat(pred_scores_list, dim=0)
                    # rois.append(all_boxes)
                    # import pdb;pdb.set_trace()
                    # 可以添加NMS等后处理 # 注意: boxes的格式应该是 [x, y, z, dx, dy, dz, heading]
                    from ....ops.iou3d_nms import iou3d_nms_utils
                    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(all_boxes[:, :7], all_boxes[:, :7])
                    nms_thresh = self.model_cfg.FEATURE_KD.ROI_POOL.get('NMS_THRESH', 0.1)

                    # 按照分数排序
                    scores_sorted, indices = torch.sort(all_scores, descending=True)
                    boxes_sorted = all_boxes[indices]

                    keep_indices = []
                    while indices.numel() > 0:
                        # 保留当前最高分数的框
                        keep_indices.append(indices[0])
                        if indices.numel() == 1:
                            break

                        # 计算当前最高分数框与其他框的IoU
                        cur_ious = iou_matrix[indices[0], indices[1:]]
                        # 找出IoU小于阈值的框
                        mask = cur_ious < nms_thresh
                        indices = indices[1:][mask]

                    # 获取保留的框
                    keep_indices = torch.stack(keep_indices)
                    nms_boxes = all_boxes[keep_indices]

                    # 如果需要限制ROI的数量
                    max_rois = self.model_cfg.FEATURE_KD.ROI_POOL.get('MAX_ROIS', 100)
                    if len(nms_boxes) > max_rois:
                        nms_boxes = nms_boxes[:max_rois]

                    rois.append(nms_boxes)
                else:
                    rois.append(torch.zeros((0, 7)).to(feature_stu.device))
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'tea_5' and batch_dict['temperature'] == 0.0:
            rois = []
            normalized_weights = self.get_normalized_weights(batch_dict['adaptive_weights'], temperature=200)
            teacher_idx = normalized_weights.argmax()
            for b_idx in range(bs):
                cur_pred_tea = batch_dict[f'decoded_pred_tea{teacher_idx + 1 if teacher_idx > 0 else ""}'][b_idx]
                pred_scores = cur_pred_tea['pred_scores']
                score_mask = pred_scores > self.model_cfg.FEATURE_KD.ROI_POOL.THRESH
                rois.append(cur_pred_tea['pred_boxes'][score_mask])
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'tea':
            rois = []
            teacher_idx = 0  #0,1,2,3,4 visual
            for b_idx in range(bs):
                cur_pred_tea = batch_dict[f'decoded_pred_tea{teacher_idx + 1 if teacher_idx > 0 else ""}'][b_idx]
                pred_scores = cur_pred_tea['pred_scores']
                score_mask = pred_scores > self.model_cfg.FEATURE_KD.ROI_POOL.THRESH
                rois.append(cur_pred_tea['pred_boxes'][score_mask])
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'stu':
            pred_dict_stu = self.dense_head.forward_ret_dict['decoded_pred_dicts']
            rois = [pred_dict_stu[i]['pred_boxes'] for i in range(bs)]
        else:
            raise NotImplementedError

        feat_tea_Cs, feat_tea_32 = features_tea[0]
        if feat_tea_Cs.shape[2] == feat_height_tea:
            voxel_size_tea = self.voxel_size_tea
            feature_map_stride_tea = self.feature_map_stride_tea
        elif feat_tea_Cs.shape[2] == feat_height:
            voxel_size_tea = self.voxel_size
            feature_map_stride_tea = self.feature_map_stride
        else:
            raise NotImplementedError

        # import pdb;pdb.set_trace()
        feat_stu_Cs, feat_stu_32 = features_stu[0]  # 如果你传进来的是 tuple
        if feat_stu_Cs.shape[2] == feat_height_tea:
            voxel_size_stu = self.voxel_size_tea
            feature_map_stride_stu = self.feature_map_stride_tea
        elif feat_stu_Cs.shape[2] == feat_height:
            voxel_size_stu = self.voxel_size
            feature_map_stride_stu = self.feature_map_stride
        else:
            raise NotImplementedError

        num_rois = 0
        for roi in rois:
            num_rois += roi.shape[0]
        if num_rois == 0:
            kd_feature_loss = 0.0
        else:
            roi_feats_tea_256, roi_feats_tea_32 = [], []
            roi_feats_stu_256, roi_feats_stu_32 = [], []

            for (stu256, stu32), (tea256, tea32) in zip(features_stu, features_tea):
                roi_feats_tea_256.append(
                    self.roi_pool_func(tea256, rois, voxel_size_tea, feature_map_stride_tea)
                )
                roi_feats_tea_32.append(
                    self.roi_pool_func(tea32, rois, voxel_size_tea, feature_map_stride_tea)
                )

                roi_feats_stu_256.append(
                    self.roi_pool_func(stu256, rois, voxel_size_stu, feature_map_stride_stu)
                )
                roi_feats_stu_32.append(
                    self.roi_pool_func(stu32, rois, voxel_size_stu, feature_map_stride_stu)
                )

            # -------------------normalized_weights---------------------------------------------
            assert len(roi_feats_tea_256) == len(batch_dict['adaptive_weights'])
            normalized_weights = self.get_normalized_weights(batch_dict['adaptive_weights'], temperature=200)
            if batch_dict['temperature'] == 0:
                max_index = normalized_weights.argmax()  # 找到索引
                normalized_weights = torch.zeros_like(normalized_weights)
                normalized_weights[max_index] = 1  # 将找到的位置设置为1
            else:
                normalized_weights = self.get_normalized_weights(batch_dict['adaptive_weights'],
                                                                 temperature=batch_dict['temperature'])

            kd_feature_loss = loss_cfg.weight * sum([normalw * self.kd_feature_loss_func(roi_feat_stu, roi_feat_tea).mean()
                               for normalw, roi_feat_stu, roi_feat_tea in zip(normalized_weights, roi_feats_stu_256, roi_feats_tea_256)])

            for tea_idx in range(len(roi_feats_tea_32)):
                roi_feats_tea = roi_feats_tea_32[tea_idx]  # [5* (400, 384, 7, 7)]
                roi_feats = roi_feats_stu_32[tea_idx]

                if loss_cfg.get('GID_ANG', None): 
                    # kd_feature_loss = 0.0
                    cnt = 0
                    # ------------------23/05/29/ 这里改成ang 加上 featur loss  --------
                    kd_feat_ang_loss = 0
                    # import pdb; pdb.set_trace()
                    for b_roi in rois:
                        num_roi = (b_roi[:, 3] != 0).sum()  # 当前图片有效ROI数量
                        if num_roi == 0:
                            continue
                        #  获取当前图片学生ROI特征cur_roi_feats 和老师ROI特征cur_roi_feats_tea
                        cur_roi_feats = roi_feats[cnt:cnt + num_roi].contiguous().view(num_roi, -1)
                        cur_roi_feats_tea = roi_feats_tea[cnt:cnt + num_roi].contiguous().view(num_roi, -1)

                        # import pdb; pdb.set_trace()
                        ang_tea = common_utils.pair_angle_gpu_sampled(cur_roi_feats_tea)
                        ang_stu = common_utils.pair_angle_gpu_sampled(cur_roi_feats)
                        kd_feat_ang_loss += torch.nn.functional.smooth_l1_loss(ang_tea, ang_stu)
                        cnt += num_roi
                    kd_feature_loss += normalized_weights[tea_idx] * (loss_cfg.ang_weight * kd_feat_ang_loss / bs)

        return kd_feature_loss
        
    def get_feature_kd_loss_spatial(self, batch_dict, loss_cfg):
        feature_name = self.model_cfg.FEATURE_KD.FEATURE_NAME
        feature_stu = batch_dict[feature_name]
        feature_name_tea = self.model_cfg.FEATURE_KD.get('FEATURE_NAME_TEA', feature_name)
        feature_tea = batch_dict[feature_name_tea + '_tea']

        if self.model_cfg.FEATURE_KD.ALIGN.target == 'student':
            target_dicts = self.dense_head.forward_ret_dict['target_dicts']
        else:
            target_dicts = batch_dict['target_dicts_tea']

        if feature_stu.shape != feature_tea.shape and self.model_cfg.FEATURE_KD.get('ALIGN', None):
            feature_tea, feature_stu = self.align_feature_map(
                feature_tea, feature_stu, align_cfg=self.model_cfg.FEATURE_KD.ALIGN
            )

        spatial_mask = kd_utils.cal_spatial_attention_mask(feature_stu)
        spatial_mask_tea = kd_utils.cal_spatial_attention_mask(feature_tea)

        # whole feature map mimicking
        bs, channel, height, width = feature_tea.shape
        feature_mask = torch.ones([bs, height, width], dtype=torch.float32).cuda()
        if loss_cfg.get('fg_mask', None):
            fg_mask = self.cal_fg_mask_from_target_heatmap_batch(target_dict=target_dicts)[0]
            feature_mask *= fg_mask

        if loss_cfg.get('score_mask', None):
            score_mask = self.cal_score_mask_from_teacher_pred(batch_dict['pred_tea'], loss_cfg.score_thresh)[0]
            feature_mask *= score_mask

        # self.feature_dist.update(kd_vis_utils.cal_feature_dist(feature_stu, feature_tea).item())
        # self.feature_dist_top100.update(kd_vis_utils.cal_feature_dist(feature_stu, feature_tea, topk=100).item())
        # self.feature_dist_spatial.update(kd_vis_utils.cal_feature_dist(spatial_mask, spatial_mask_tea).item())

        # calculate spatial magnitute inside objects and non-empty regions
        # non_empty_mask = torch.zeros(feature_stu.shape[2:]).cuda()
        # voxel_coords = batch_dict['voxel_coords'].long()
        # non_empty_mask[voxel_coords[:, 2], voxel_coords[:, -1]] = True

        # self.spatial_fg_meter.update(spatial_mask[fg_mask.bool()].mean())
        # self.spatial_nonempty_meter.update(spatial_mask[non_empty_mask.unsqueeze(0).bool()].mean())
        # self.spatial_all_meter.update(spatial_mask.mean())

        kd_feature_loss_all = self.kd_feature_loss_func(spatial_mask, spatial_mask_tea)
        kd_feature_loss = (kd_feature_loss_all * feature_mask).sum() / (feature_mask.sum() + 1e-6)

        kd_feature_loss = kd_feature_loss * loss_cfg.weight

        return kd_feature_loss
    
    def get_feature_kd_loss_affinity(self, batch_dict, loss_cfg):
        feature_name = self.model_cfg.FEATURE_KD.FEATURE_NAME
        feature_stu = batch_dict[feature_name]
        feature_name_tea = self.model_cfg.FEATURE_KD.get('FEATURE_NAME_TEA', feature_name)
        feature_tea = batch_dict[feature_name_tea + '_tea']

        feat_height = feature_stu.shape[2]
        feat_height_tea = feature_tea.shape[2]

        bs, ch = feature_stu.shape[0], feature_stu.shape[1]
        if self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'gt':
            rois = batch_dict['gt_boxes'].detach()
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'tea':
            rois = []
            for b_idx in range(bs):
                cur_pred_tea = batch_dict['decoded_pred_tea'][b_idx]
                pred_scores = cur_pred_tea['pred_scores']
                score_mask = pred_scores > self.model_cfg.FEATURE_KD.ROI_POOL.THRESH
                rois.append(cur_pred_tea['pred_boxes'][score_mask])
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'stu':
            pred_dict_stu = self.dense_head.forward_ret_dict['decoded_pred_dicts']
            rois = [pred_dict_stu[i]['pred_boxes'] for i in range(bs)]
        else:
            raise NotImplementedError

        if feature_stu.shape[2] == feat_height_tea:
            voxel_size_stu = self.voxel_size_tea
            feature_map_stride_stu = self.feature_map_stride_tea
        elif feature_stu.shape[2] == feat_height:
            voxel_size_stu = self.voxel_size
            feature_map_stride_stu = self.feature_map_stride
        else:
            raise NotImplementedError

        if feature_tea.shape[2] == feat_height_tea:
            voxel_size_tea = self.voxel_size_tea
            feature_map_stride_tea = self.feature_map_stride_tea
        elif feature_tea.shape[2] == feat_height:
            voxel_size_tea = self.voxel_size
            feature_map_stride_tea = self.feature_map_stride
        else:
            raise NotImplementedError

        roi_feats = self.roi_pool_func(
            feature_stu, rois, voxel_size_stu, feature_map_stride_stu
        )
        roi_feats_tea = self.roi_pool_func(
            feature_tea, rois, voxel_size_tea, feature_map_stride_tea
        )

        # calculate intro object affinity
        intra_aff_matrix = self.cal_cos_sim_affinity_matrix(roi_feats.view(roi_feats.shape[0], ch, -1))
        intra_aff_matrix_tea = self.cal_cos_sim_affinity_matrix(roi_feats_tea.view(roi_feats.shape[0], ch, -1))

        kd_feature_loss = loss_cfg.weight * self.kd_feature_loss_func(
            intra_aff_matrix, intra_aff_matrix_tea
        ).mean()

        return kd_feature_loss

    @staticmethod
    def cal_cos_sim_affinity_matrix(roi_features):
        """_summary_

        Args:
            roi_features (_type_): [N, C, K]
        """
        # [N, K, K]
        sim_matrix = torch.matmul(roi_features.transpose(1, 2), roi_features)
        norm = torch.norm(roi_features, dim=1, keepdim=True)
        affinity_matrix = sim_matrix / torch.clamp((norm * norm.transpose(1, 2)), min=1e-6)

        return affinity_matrix
