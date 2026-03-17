import random

import numpy.ma
import torch

from pcdet.models.kd_heads.kd_head import KDHeadTemplate
from pcdet.utils import loss_utils


class CenterLogitKDHead(KDHeadTemplate):
    def __init__(self, model_cfg, dense_head):
        super().__init__(model_cfg, dense_head)

    def build_logit_kd_loss(self):
        # logit kd hm loss
        if self.model_cfg.KD_LOSS.HM_LOSS.type in ['FocalLossCenterNet']:
            self.kd_hm_loss_func = getattr(loss_utils, self.model_cfg.KD_LOSS.HM_LOSS.type)(
                pos_thresh=self.model_cfg.KD_LOSS.HM_LOSS.pos_thresh
            )
        elif self.model_cfg.KD_LOSS.HM_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
            self.kd_hm_loss_func = getattr(torch.nn, self.model_cfg.KD_LOSS.HM_LOSS.type)(reduction='none')
        else:
            raise NotImplementedError

        # logit kd hm_sort loss
        if self.model_cfg.KD_LOSS.get('HM_SORT_LOSS', None):
            self.kd_hm_sort_loss_func = loss_utils.SortLoss(rank=self.model_cfg.KD_LOSS.HM_SORT_LOSS.rank)
        else:
            self.kd_hm_sort_loss_func = None

        # logit kd regression loss
        if self.model_cfg.KD_LOSS.REG_LOSS.type == 'WeightedSmoothL1Loss':
            self.kd_reg_loss_func = getattr(loss_utils, self.model_cfg.KD_LOSS.REG_LOSS.type)(
                code_weights=self.model_cfg.KD_LOSS.reg_loss.code_weights
            )
        elif self.model_cfg.KD_LOSS.REG_LOSS.type == 'RegLossCenterNet':
            self.kd_reg_loss_func = getattr(loss_utils, self.model_cfg.KD_LOSS.REG_LOSS.type)()
        else:
            raise NotImplementedError

    def get_logit_kd_loss(self, batch_dict, tb_dict):
        if self.model_cfg.LOGIT_KD.MODE == 'decoded_boxes':
            pred_tea = batch_dict['decoded_pred_tea']
            kd_logit_loss, kd_hm_loss, kd_reg_loss = self.get_kd_loss_with_decoded_boxes(
                pred_tea, self.model_cfg.KD_LOSS
            )
        elif self.model_cfg.LOGIT_KD.MODE == 'raw_pred':
            pred_tea = batch_dict['pred_tea']
            if self.model_cfg.KD_LOSS.MKD == 'hm' and (batch_dict.get('pred_tea2') is not None):
                kd_logit_loss, kd_hm_loss, kd_reg_loss, kd_sort_loss = self.get_kd_loss_with_raw_prediction2(
                    pred_tea, batch_dict['pred_tea2'], self.model_cfg.KD_LOSS,
                    target_dict_tea=batch_dict['target_dicts_tea'],
                    target_dict_tea2=batch_dict['target_dicts_tea2']
                )
            elif self.model_cfg.KD_LOSS.MKD == 'random' and (batch_dict.get('pred_tea2') is not None):
                i = random.randint(1,2)
                # import pdb; pdb.set_trace()
                if i == 1:
                    kd_logit_loss, kd_hm_loss, kd_reg_loss, kd_sort_loss = self.get_kd_loss_with_raw_prediction(
                        pred_tea, self.model_cfg.KD_LOSS, target_dict_tea=batch_dict['target_dicts_tea']
                    )
                elif i==2:
                    pred_tea2 = batch_dict['pred_tea2']
                    kd_logit_loss, kd_hm_loss, kd_reg_loss, kd_sort_loss = self.get_kd_loss_with_raw_prediction(
                        pred_tea2, self.model_cfg.KD_LOSS, target_dict_tea=batch_dict['target_dicts_tea2']
                    )
            elif self.model_cfg.KD_LOSS.MKD == 'adaptive' and (batch_dict.get('pred_tea5') is not None):  #--tea_5-----
                pred_teas = [
                    pred_tea,batch_dict['pred_tea2'], batch_dict['pred_tea3'],batch_dict['pred_tea4'],batch_dict['pred_tea5']
                ]
                target_dict_teas = [
                    batch_dict['target_dicts_tea'],batch_dict['target_dicts_tea2'],batch_dict['target_dicts_tea3'],
                    batch_dict['target_dicts_tea4'],batch_dict['target_dicts_tea5']
                ]
                kd_logit_loss, kd_hm_loss, kd_reg_loss, kd_sort_loss, adaptive_weights = self.get_kd_loss_with_raw_prediction_gt5(
                    pred_teas, batch_dict['gt_hm'], self.model_cfg.KD_LOSS,
                    target_dict_teas, softhm=self.model_cfg.KD_LOSS.softhm, temperature=batch_dict['temperature']
                )
                batch_dict['adaptive_weights'] = adaptive_weights
            elif self.model_cfg.KD_LOSS.MKD == 'adaptive' and (batch_dict.get('pred_tea2') is not None):
                kd_logit_loss, kd_hm_loss, kd_reg_loss, kd_sort_loss = self.get_kd_loss_with_raw_prediction_gt(
                    pred_tea, batch_dict['pred_tea2'], batch_dict['gt_hm'], self.model_cfg.KD_LOSS,
                    target_dict_tea=batch_dict['target_dicts_tea'],
                    target_dict_tea2=batch_dict['target_dicts_tea2'],
                    softhm=self.model_cfg.KD_LOSS.softhm,
                    temperature=batch_dict['temperature']
                )
            elif batch_dict.get('pred_tea') is not None:
                kd_logit_loss, kd_hm_loss, kd_reg_loss, kd_sort_loss = self.get_kd_loss_with_raw_prediction(
                    pred_tea, self.model_cfg.KD_LOSS, target_dict_tea=batch_dict['target_dicts_tea']
                )
                if batch_dict.get('pred_tea2') is not None:
                    pred_tea2 = batch_dict['pred_tea2']
                    kd_logit_loss2, kd_hm_loss2, kd_reg_loss2, kd_sort_loss2 = self.get_kd_loss_with_raw_prediction(
                        pred_tea2, self.model_cfg.KD_LOSS, target_dict_tea=batch_dict['target_dicts_tea2']
                    )
                    kd_logit_loss, kd_hm_loss, kd_reg_loss, kd_sort_loss = (
                        kd_logit_loss + kd_logit_loss2,
                        kd_hm_loss + kd_hm_loss2,
                        kd_reg_loss + kd_reg_loss2,
                        kd_sort_loss + kd_sort_loss2
                    )

        elif self.model_cfg.LOGIT_KD.MODE == 'target':
            kd_logit_loss, kd_hm_loss, kd_reg_loss = self.get_kd_loss_with_target_tea(
                batch_dict['pred_tea'], self.model_cfg.KD_LOSS, target_dict_tea=batch_dict['target_dicts_tea']
            )
        else:
            raise NotImplementedError


        tb_dict['kd_hm_ls'] = kd_hm_loss if isinstance(kd_hm_loss, float) else kd_hm_loss.item()
        tb_dict['kd_loc_ls'] = kd_reg_loss if isinstance(kd_reg_loss, float) else kd_reg_loss.item()
        if batch_dict.get('pred_tea5') is not None:
            tb_dict['temp5'] = batch_dict['temperature']
        return kd_logit_loss, tb_dict

    def get_kd_loss_with_raw_prediction(self, pred_tea, loss_cfg, target_dict_tea):
        """
        Args:
            pred_tea: pred_dict of teacher
                center: [bs, 2, feat_h, feat_w]. Offset to the nearest center
                center_z: [bs, 1, feat_h, feat_w]. absolute coordinates


            loss_cfg: kd loss config

        Returns:

        """
        pred_stu = self.dense_head.forward_ret_dict['pred_dicts']
        if self.model_cfg.LOGIT_KD.ALIGN.target == 'student':
            target_dicts = self.dense_head.forward_ret_dict['target_dicts']
        else:
            target_dicts = target_dict_tea

        assert len(pred_tea) == len(pred_stu)

        kd_hm_loss = 0
        kd_reg_loss = 0

        for idx, cur_pred_stu in enumerate(pred_stu):
            cur_pred_tea = pred_tea[idx]
            cur_hm_tea = self.dense_head.sigmoid(cur_pred_tea['hm'])

            # interpolate if needed
            if (cur_hm_tea.shape != cur_pred_stu['hm'].shape) and self.model_cfg.LOGIT_KD.get('ALIGN', None):
                hm_tea, hm_stu = self.align_feature_map(
                    cur_hm_tea, cur_pred_stu['hm'], self.model_cfg.LOGIT_KD.ALIGN
                )
            else:
                hm_tea, hm_stu = cur_hm_tea, cur_pred_stu['hm']

            # classification loss
            if loss_cfg.HM_LOSS.weight == 0:
                kd_hm_loss_raw = 0
            elif loss_cfg.HM_LOSS.type == 'FocalLossCenterNet':
                if loss_cfg.HM_LOSS.get('inverse', None):
                    kd_hm_loss_raw = self.kd_hm_loss_func(hm_tea, hm_stu)
                else:
                    kd_hm_loss_raw = self.kd_hm_loss_func(hm_stu, hm_tea)
            elif loss_cfg.HM_LOSS.type == 'WeightedSmoothL1Loss':
                bs, channel = hm_stu.shape[0], hm_stu.shape[1]
                heatmap_stu = hm_stu.view(bs, channel, -1).permute(0, 2, 1)
                heatmap_tea = hm_tea.view(bs, channel, -1).permute(0, 2, 1)
                kd_hm_loss_all = self.kd_hm_loss_func(heatmap_stu, heatmap_tea)
                # position-wise confidence mask: shape [bs, h*w, c]
                mask = (torch.max(heatmap_stu, -1)[0] > loss_cfg.HM_LOSS.thresh).float() * \
                                  (torch.max(heatmap_tea, -1)[0] > loss_cfg.HM_LOSS.thresh).float()
                kd_hm_loss_raw = (kd_hm_loss_all * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-6)
            elif loss_cfg.HM_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
                kd_hm_loss_all = self.kd_hm_loss_func(hm_stu, hm_tea)
                mask = (torch.max(hm_tea, dim=1)[0] > loss_cfg.HM_LOSS.thresh).float()
                if loss_cfg.HM_LOSS.get('fg_mask', None):
                    fg_mask = self.cal_fg_mask_from_target_heatmap_batch(
                        target_dicts, soft=loss_cfg.HM_LOSS.get('soft_mask', None)
                    )[idx]
                    mask *= fg_mask

                if loss_cfg.HM_LOSS.get('rank', -1) != -1:
                    rank_mask = self.cal_rank_mask_from_teacher_pred(pred_tea, K=loss_cfg.HM_LOSS.rank)[idx]
                    mask *= rank_mask

                kd_hm_loss_raw = (kd_hm_loss_all * mask.unsqueeze(1)).sum() / torch.count_nonzero(mask)
            else:
                raise NotImplementedError
            kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw

            if self.kd_hm_sort_loss_func is not None and loss_cfg.HM_SORT_LOSS.weight != 0:
                kd_hm_sort_loss = self.kd_hm_sort_loss_func(hm_stu, hm_tea)
                kd_hm_sort_loss = loss_cfg.HM_SORT_LOSS.weight * kd_hm_sort_loss
            else:
                kd_hm_sort_loss = 0.0

            # localization loss
            # parse teacher prediction to target style
            pred_boxes_tea = torch.cat([cur_pred_tea[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)
            pred_boxes_stu = torch.cat([cur_pred_stu[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)
            if loss_cfg.REG_LOSS.weight == 0 or (pred_boxes_tea.shape != pred_boxes_stu.shape):
                kd_reg_loss_raw = 0
            elif loss_cfg.REG_LOSS.type == 'RegLossCenterNet':
                pred_boxes_tea_selected = loss_utils._transpose_and_gather_feat(pred_boxes_tea, target_dicts['inds'][idx])

                kd_reg_loss_raw = self.kd_reg_loss_func(
                    pred_boxes_stu, target_dicts['masks'][idx], target_dicts['inds'][idx], pred_boxes_tea_selected
                )
                kd_reg_loss_raw = (kd_reg_loss_raw * kd_reg_loss_raw.new_tensor(
                    loss_cfg.REG_LOSS.code_weights)).sum()
            else:
                raise NotImplementedError
            kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw

        kd_loss = (kd_hm_loss + kd_hm_sort_loss + kd_reg_loss) / len(pred_stu)

        return kd_loss, kd_hm_loss / len(pred_stu), kd_reg_loss / len(pred_stu), kd_hm_sort_loss / len(pred_stu)

    def get_kd_loss_with_raw_prediction2(self, pred_tea, pred_tea2, loss_cfg, target_dict_tea, target_dict_tea2):
        """
        Args:
            pred_tea: pred_dict of teacher
                center: [bs, 2, feat_h, feat_w]. Offset to the nearest center
                center_z: [bs, 1, feat_h, feat_w]. absolute coordinates
            loss_cfg: kd loss config
        Returns:
        """
        pred_stu = self.dense_head.forward_ret_dict['pred_dicts']
        if self.model_cfg.LOGIT_KD.ALIGN.target == 'student':
            target_dicts = self.dense_head.forward_ret_dict['target_dicts']
        elif self.model_cfg.LOGIT_KD.ALIGN.target == 'teacher2':
            target_dicts = target_dict_tea2
        else:
            target_dicts = target_dict_tea

        assert len(pred_tea) == len(pred_stu)
        assert len(pred_tea2) == len(pred_stu)

        kd_hm_loss = 0
        kd_reg_loss = 0

        for idx, cur_pred_stu in enumerate(pred_stu):
            cur_pred_tea = pred_tea[idx]
            cur_pred_tea2 = pred_tea2[idx]
            cur_hm_tea = self.dense_head.sigmoid((cur_pred_tea['hm']+cur_pred_tea2['hm'])/2)  # MKD:hm 取平均

            # interpolate if needed
            if (cur_hm_tea.shape != cur_pred_stu['hm'].shape) and self.model_cfg.LOGIT_KD.get('ALIGN', None):
                hm_tea, hm_stu = self.align_feature_map(
                    cur_hm_tea, cur_pred_stu['hm'], self.model_cfg.LOGIT_KD.ALIGN
                )
            else:
                hm_tea, hm_stu = cur_hm_tea, cur_pred_stu['hm']

            # classification loss
            if loss_cfg.HM_LOSS.weight == 0:
                kd_hm_loss_raw = 0
            elif loss_cfg.HM_LOSS.type == 'FocalLossCenterNet':
                if loss_cfg.HM_LOSS.get('inverse', None):
                    kd_hm_loss_raw = self.kd_hm_loss_func(hm_tea, hm_stu)
                else:
                    kd_hm_loss_raw = self.kd_hm_loss_func(hm_stu, hm_tea)
            elif loss_cfg.HM_LOSS.type == 'WeightedSmoothL1Loss':
                bs, channel = hm_stu.shape[0], hm_stu.shape[1]
                heatmap_stu = hm_stu.view(bs, channel, -1).permute(0, 2, 1)
                heatmap_tea = hm_tea.view(bs, channel, -1).permute(0, 2, 1)
                kd_hm_loss_all = self.kd_hm_loss_func(heatmap_stu, heatmap_tea)
                # position-wise confidence mask: shape [bs, h*w, c]
                mask = (torch.max(heatmap_stu, -1)[0] > loss_cfg.HM_LOSS.thresh).float() * \
                       (torch.max(heatmap_tea, -1)[0] > loss_cfg.HM_LOSS.thresh).float()
                kd_hm_loss_raw = (kd_hm_loss_all * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-6)
            elif loss_cfg.HM_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
                kd_hm_loss_all = self.kd_hm_loss_func(hm_stu, hm_tea)
                # position-wise confidence mask: shape [bs, c, h, w]
                mask = (torch.max(hm_tea, dim=1)[0] > loss_cfg.HM_LOSS.thresh).float()


                if loss_cfg.HM_LOSS.get('fg_mask', None):
                    fg_mask = self.cal_fg_mask_from_target_heatmap_batch(
                        target_dicts, soft=loss_cfg.HM_LOSS.get('soft_mask', None)
                    )[idx]
                    mask *= fg_mask

                if loss_cfg.HM_LOSS.get('rank', -1) != -1:
                    rank_mask = self.cal_rank_mask_from_teacher_pred(pred_tea, K=loss_cfg.HM_LOSS.rank)[idx]
                    mask *= rank_mask

                kd_hm_loss_raw = (kd_hm_loss_all * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-6)

            else:
                raise NotImplementedError
            kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw

            if self.kd_hm_sort_loss_func is not None and loss_cfg.HM_SORT_LOSS.weight != 0:
                kd_hm_sort_loss = self.kd_hm_sort_loss_func(hm_stu, hm_tea)
                kd_hm_sort_loss = loss_cfg.HM_SORT_LOSS.weight * kd_hm_sort_loss
            else:
                kd_hm_sort_loss = 0.0

            # localization loss
            # parse teacher prediction to target style
            pred_boxes_tea = torch.cat(
                [cur_pred_tea[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)
            # -------- tea_2 -------------------
            pred_boxes_tea2 = torch.cat(
                [cur_pred_tea2[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)
            pred_boxes_stu = torch.cat(
                [cur_pred_stu[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)
            if loss_cfg.REG_LOSS.weight == 0 or (pred_boxes_tea.shape != pred_boxes_stu.shape):
                kd_reg_loss_raw = 0
            elif loss_cfg.REG_LOSS.type == 'RegLossCenterNet':
                pred_boxes_tea_selected = loss_utils._transpose_and_gather_feat(pred_boxes_tea,
                                                                                target_dicts['inds'][idx])
                pred_boxes_tea_selected2 = loss_utils._transpose_and_gather_feat(pred_boxes_tea2,
                                                                                target_dict_tea2['inds'][idx])
                kd_reg_loss_raw = self.kd_reg_loss_func(
                    pred_boxes_stu, target_dicts['masks'][idx], target_dicts['inds'][idx], pred_boxes_tea_selected
                )
                kd_reg_loss_raw2 = self.kd_reg_loss_func(
                    pred_boxes_stu, target_dict_tea2['masks'][idx], target_dict_tea2['inds'][idx], pred_boxes_tea_selected2
                )
                kd_reg_loss_raw = ((kd_reg_loss_raw+kd_reg_loss_raw2)/2 * kd_reg_loss_raw.new_tensor(
                    loss_cfg.REG_LOSS.code_weights)).sum()  # 待改进是优化系数pred
            else:
                raise NotImplementedError
            kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw

        kd_loss = (kd_hm_loss + kd_hm_sort_loss + kd_reg_loss) / len(pred_stu)

        return kd_loss, kd_hm_loss / len(pred_stu), kd_reg_loss / len(pred_stu), kd_hm_sort_loss / len(pred_stu)

    def get_sample_kd_loss_hm(self,hm_tea, hm_stu, loss_cfg, target_dicts, idx):
        # classification loss
        if loss_cfg.HM_LOSS.weight == 0:
            kd_hm_loss_raw = 0
        elif loss_cfg.HM_LOSS.type == 'FocalLossCenterNet':
            if loss_cfg.HM_LOSS.get('inverse', None):
                kd_hm_loss_raw = self.kd_hm_loss_func(hm_tea, hm_stu)
            else:
                kd_hm_loss_raw = self.kd_hm_loss_func(hm_stu, hm_tea)
        elif loss_cfg.HM_LOSS.type == 'WeightedSmoothL1Loss':
            bs, channel = hm_stu.shape[0], hm_stu.shape[1]
            heatmap_stu = hm_stu.view(bs, channel, -1).permute(0, 2, 1)
            heatmap_tea = hm_tea.view(bs, channel, -1).permute(0, 2, 1)
            kd_hm_loss_all = self.kd_hm_loss_func(heatmap_stu, heatmap_tea)
            # position-wise confidence mask: shape [bs, h*w, c]
            mask = (torch.max(heatmap_stu, -1)[0] > loss_cfg.HM_LOSS.thresh).float() * \
                   (torch.max(heatmap_tea, -1)[0] > loss_cfg.HM_LOSS.thresh).float()
            kd_hm_loss_raw = (kd_hm_loss_all * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-6)
        elif loss_cfg.HM_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
            kd_hm_loss_all = self.kd_hm_loss_func(hm_stu, hm_tea)
            # position-wise confidence mask: shape [bs, c, h, w]
            mask = (torch.max(hm_tea, dim=1)[0] > loss_cfg.HM_LOSS.thresh).float()
            if loss_cfg.HM_LOSS.get('soft_mask', None):
                mask = torch.max(hm_tea, dim=1)[0] * mask

            kd_hm_loss_raw = (kd_hm_loss_all * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-6)
        else:
            raise NotImplementedError
        return kd_hm_loss_raw

    def get_sample_kd_loss_reg(self, loss_cfg, cur_pred_tea, cur_pred_stu, target_dicts, idx):
        pred_boxes_tea = torch.cat(
            [cur_pred_tea[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)
        pred_boxes_stu = torch.cat(
            [cur_pred_stu[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)
        if loss_cfg.REG_LOSS.weight == 0 or (pred_boxes_tea.shape != pred_boxes_stu.shape):
            kd_reg_loss_raw = 0
        elif loss_cfg.REG_LOSS.type == 'RegLossCenterNet':
            pred_boxes_tea_selected = loss_utils._transpose_and_gather_feat(pred_boxes_tea,
                                                                            target_dicts['inds'][idx])

            kd_reg_loss_raw = self.kd_reg_loss_func(
                pred_boxes_stu, target_dicts['masks'][idx], target_dicts['inds'][idx], pred_boxes_tea_selected
            )
            kd_reg_loss_raw = (kd_reg_loss_raw * kd_reg_loss_raw.new_tensor(
                loss_cfg.REG_LOSS.code_weights)).sum()
        else:
            raise NotImplementedError
        # kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw
        return kd_reg_loss_raw

    def get_normalized_weights(self, adaptive_weights, temperature=2.0, adaptive_mask=None):
        tensor_weights = torch.tensor(adaptive_weights, dtype=torch.float32)
        if adaptive_mask is None:
            adaptive_mask = torch.ones_like(tensor_weights)

        softmax_weights = torch.nn.functional.softmax(-tensor_weights / temperature, dim=0)  # 行方向(dim=0)

        return softmax_weights
    def get_confidence_weights(self, adaptive_weights, temperature=2.0, adaptive_mask=None):
        tensor_weights = torch.tensor(adaptive_weights, dtype=torch.float32)
        if adaptive_mask is None:
            adaptive_mask = torch.ones_like(tensor_weights)

        sum_weights = (adaptive_mask*torch.exp(tensor_weights / temperature)).sum()
        softmax_weights = adaptive_mask * torch.exp(tensor_weights / temperature) / sum_weights
        softmax_weights = (1 - softmax_weights) /(adaptive_mask.sum()-1)
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
    def min_max_normalize(self, adaptive_weights, adaptive_mask=None):
        """
        Perform min-max normalization on a list of values.
        Args:
            adaptive_weights (list): A list of floats.
        Returns:
            list: A list of normalized floats.
        """
        # Convert to a PyTorch tensor
        tensor_weights = torch.tensor(adaptive_weights, dtype=torch.float32)
        if adaptive_mask is None:
            adaptive_mask= torch.ones_like(tensor_weights)
        tensor_weights = tensor_weights * adaptive_mask
        # Min-Max normalization: (x - min) / (max - min)
        min_val = torch.min(tensor_weights[adaptive_mask > 0])
        max_val = torch.max(tensor_weights[adaptive_mask > 0])
        # Prevent division by zero
        if (max_val - min_val) == 0:
            return tensor_weights / tensor_weights.sum()
        normalized_weights = (tensor_weights - min_val) / (max_val - min_val)
        # Convert to list
        return normalized_weights
    def TEA_ACCUR_normalize(self, TEA_ACCUR_weights, adaptive_mask=None):
        tensor_weights = torch.tensor(TEA_ACCUR_weights, dtype=torch.float32)
        if adaptive_mask is None:
            adaptive_mask= torch.ones_like(tensor_weights)
        tensor_weights = tensor_weights * adaptive_mask
        normalized_weights = tensor_weights/tensor_weights.sum()
        return normalized_weights
    def get_kd_loss_with_raw_prediction_gt(self, pred_tea, pred_tea2, gt_labels, loss_cfg, target_dict_tea, target_dict_tea2, softhm=None, temperature=None):
        """
        Args:
            pred_tea: pred_dict of teacher
                center: [bs, 2, feat_h, feat_w]. Offset to the nearest center
                center_z: [bs, 1, feat_h, feat_w]. absolute coordinates
            loss_cfg: kd loss config
        Returns:
        """
        pred_stu = self.dense_head.forward_ret_dict['pred_dicts']
        if self.model_cfg.LOGIT_KD.ALIGN.target == 'student':
            target_dicts = self.dense_head.forward_ret_dict['target_dicts']
        elif self.model_cfg.LOGIT_KD.ALIGN.target == 'teacher2':
            target_dicts = target_dict_tea2
        else:
            target_dicts = target_dict_tea

        assert len(pred_tea) == len(pred_stu)
        assert len(pred_tea2) == len(pred_stu)
        assert len(gt_labels) == len(pred_stu)

        kd_hm_loss = 0
        kd_reg_loss = 0

        for idx, cur_pred_stu in enumerate(pred_stu):
            cur_pred_tea = pred_tea[idx]
            cur_pred_tea2 = pred_tea2[idx]
            # import pdb;pdb.set_trace()
            cur_pred_gt = gt_labels[idx]
            # cur_hm_tea = self.dense_head.sigmoid((cur_pred_tea['hm']+cur_pred_tea2['hm'])/2)  # MKD:hm 取平均

            # interpolate if needed
            cur_hm_tea = self.dense_head.sigmoid(cur_pred_tea['hm'])
            if (cur_hm_tea.shape != cur_pred_stu['hm'].shape) and self.model_cfg.LOGIT_KD.get('ALIGN', None):
                hm_tea, hm_stu = self.align_feature_map(
                    cur_hm_tea, cur_pred_stu['hm'], self.model_cfg.LOGIT_KD.ALIGN
                )
            else:
                hm_tea, hm_stu = cur_hm_tea, cur_pred_stu['hm']

            cur_hm_tea2 = self.dense_head.sigmoid(cur_pred_tea2['hm'])
            if (cur_hm_tea2.shape != cur_pred_stu['hm'].shape) and self.model_cfg.LOGIT_KD.get('ALIGN', None):
                hm_tea2, hm_stu = self.align_feature_map(
                    cur_hm_tea2, cur_pred_stu['hm'], self.model_cfg.LOGIT_KD.ALIGN
                )
            else:
                hm_tea2, hm_stu = cur_hm_tea2, cur_pred_stu['hm']

            if (cur_pred_gt.shape != cur_pred_stu['hm'].shape) and self.model_cfg.LOGIT_KD.get('ALIGN', None):
                hm_gt, hm_stu = self.align_feature_map(
                    cur_pred_gt, cur_pred_stu['hm'], self.model_cfg.LOGIT_KD.ALIGN
                )
            else:
                hm_gt, hm_stu = cur_pred_gt, cur_pred_stu['hm']


            mask = (torch.max(hm_gt, dim=1)[0] > loss_cfg.HM_LOSS.thresh).float().unsqueeze(1)
            adaptive_weight1 = (100.0 * (self.kd_hm_loss_func(hm_tea, hm_gt) * mask).sum() / mask.sum()).clamp(min=0.1, max=5.0)
            adaptive_weight2 = (100.0 * (self.kd_hm_loss_func(hm_tea2, hm_gt) * mask).sum() / mask.sum()).clamp(min=0.1, max=5.0)
            adaptive_weights = [adaptive_weight1, adaptive_weight2]
            adaptive_mask = self.create_softmax_mask(adaptive_weights, max_thred=5.0, min_thred=0.1)
            if temperature == 0.0:
                softhm = 'single'
            if softhm == 'single':
                hm_tea = hm_tea
                target_dicts = target_dict_tea
                kd_hm_loss_raw = self.get_sample_kd_loss_hm(hm_tea, hm_stu, loss_cfg, target_dicts, idx)
                kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw
                kd_reg_loss += loss_cfg.REG_LOSS.weight * self.get_sample_kd_loss_reg(loss_cfg, cur_pred_tea,
                                                                                      cur_pred_stu, target_dicts, idx)
            elif softhm =='select':
                if adaptive_weight1 > adaptive_weight2:
                    hm_tea = hm_tea2
                    target_dicts = target_dict_tea2
                else:
                    hm_tea = hm_tea
                    target_dicts = target_dict_tea
                kd_hm_loss_raw = self.get_sample_kd_loss_hm(hm_tea, hm_stu, loss_cfg, target_dicts, idx)
                kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw
                kd_reg_loss += loss_cfg.REG_LOSS.weight * self.get_sample_kd_loss_reg(loss_cfg, cur_pred_tea, cur_pred_stu, target_dicts, idx)
            elif softhm=='addition':
                kd_hm_loss_raw = (self.get_sample_kd_loss_hm(hm_tea, hm_stu, loss_cfg, target_dict_tea, idx)+self.get_sample_kd_loss_hm(hm_tea2, hm_stu, loss_cfg, target_dict_tea2, idx))/2
                kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw
                kd_reg_loss += loss_cfg.REG_LOSS.weight * (self.get_sample_kd_loss_reg(loss_cfg, cur_pred_tea, cur_pred_stu, target_dict_tea, idx)
                             +self.get_sample_kd_loss_reg(loss_cfg, cur_pred_tea2, cur_pred_stu, target_dict_tea2, idx))/2
            elif softhm == 'Soft':
                normalized_weights = self.get_normalized_weights(adaptive_weights, temperature=temperature, adaptive_mask=adaptive_mask)
                kd_hm_loss_raw = (normalized_weights[0] * self.get_sample_kd_loss_hm(hm_tea, hm_stu,
                                                                                                loss_cfg,
                                                                                                target_dict_tea, idx)
                                  + normalized_weights[1] * self.get_sample_kd_loss_hm(hm_tea2, hm_stu,
                                                                                                  loss_cfg,
                                                                                                  target_dict_tea2,
                                                                                                  idx)) #\
                kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw
                kd_reg_loss_raw = (normalized_weights[0] * self.get_sample_kd_loss_reg(loss_cfg,
                                                                                                  cur_pred_tea,
                                                                                                  cur_pred_stu,
                                                                                                  target_dict_tea, idx)
                                   + normalized_weights[1] * self.get_sample_kd_loss_reg(loss_cfg,
                                                                                                    cur_pred_tea2,
                                                                                                    cur_pred_stu,
                                                                                                    target_dict_tea2,
                                                                                                    idx)) #\
                kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw

            if self.kd_hm_sort_loss_func is not None and loss_cfg.HM_SORT_LOSS.weight != 0:
                kd_hm_sort_loss = self.kd_hm_sort_loss_func(hm_stu, hm_tea)
                kd_hm_sort_loss = loss_cfg.HM_SORT_LOSS.weight * kd_hm_sort_loss
            else:
                kd_hm_sort_loss = 0.0
        kd_loss = (kd_hm_loss + kd_hm_sort_loss + kd_reg_loss) / len(pred_stu)
        return kd_loss, kd_hm_loss / len(pred_stu), kd_reg_loss / len(pred_stu), kd_hm_sort_loss / len(pred_stu)

    def get_kd_loss_with_raw_prediction_gt5(self, pred_teas, gt_labels, loss_cfg, target_dict_teas, softhm=None, temperature=None):
        """
        Args:
            pred_tea: pred_dict of teacher
                center: [bs, 2, feat_h, feat_w]. Offset to the nearest center
                center_z: [bs, 1, feat_h, feat_w]. absolute coordinates
            loss_cfg: kd loss config
        Returns:
        """
        pred_stu = self.dense_head.forward_ret_dict['pred_dicts']

        assert len(pred_teas) == 5
        for pred_tea in pred_teas:
            assert len(pred_tea) == len(pred_stu)
        assert len(gt_labels) == len(pred_stu)

        kd_hm_loss = 0
        kd_reg_loss = 0

        for idx, cur_pred_stu in enumerate(pred_stu):
            cur_pred_teas = [pred_tea[idx] for pred_tea in pred_teas]
            hm_teas = []

            for cur_pred_tea in cur_pred_teas:
                cur_hm_tea = self.dense_head.sigmoid(cur_pred_tea['hm'])

                # Interpolate if needed 确保size一致
                if (cur_hm_tea.shape != cur_pred_stu['hm'].shape) and self.model_cfg.LOGIT_KD.get('ALIGN', None):
                    hm_tea, _ = self.align_feature_map(
                        cur_hm_tea, cur_pred_stu['hm'], self.model_cfg.LOGIT_KD.ALIGN
                    )
                else:
                    hm_tea = cur_hm_tea
                hm_teas.append(hm_tea)

            if (gt_labels[idx].shape != cur_pred_stu['hm'].shape) and self.model_cfg.LOGIT_KD.get('ALIGN', None):
                hm_gt, hm_stu = self.align_feature_map(
                    gt_labels[idx], cur_pred_stu['hm'], self.model_cfg.LOGIT_KD.ALIGN
                )
            else:
                hm_gt, hm_stu = gt_labels[idx], cur_pred_stu['hm']

            ## Calculate adaptive weights for all teachers
            mask = (torch.max(hm_gt, dim=1)[0] > loss_cfg.HM_LOSS.thresh).float().unsqueeze(1)
            adaptive_weights = [(100.0 * (self.kd_hm_loss_func(hm_tea, hm_gt) * mask).sum() / mask.sum()) for hm_tea in hm_teas]
            # adaptive_mask = self.create_softmax_mask(adaptive_weights, max_thred=5.0, min_thred=0.1)
            if temperature == 0.0:
                softhm = 'single'
            elif temperature == 0.02:
                softhm = 'select'

            if softhm == 'single':
                # Select the teacher with minimum loss
                best_teacher_idx = 1
                hm_tea = hm_teas[best_teacher_idx]
                target_dicts = target_dict_teas[best_teacher_idx]

                kd_hm_loss_raw = self.get_sample_kd_loss_hm(hm_tea, hm_stu, loss_cfg, target_dicts, idx)
                kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw
                kd_reg_loss += loss_cfg.REG_LOSS.weight * self.get_sample_kd_loss_reg(
                    loss_cfg, cur_pred_teas[best_teacher_idx], cur_pred_stu, target_dicts, idx
                )
            elif softhm == 'select':
                # Select the teacher with minimum loss
                best_teacher_idx = adaptive_weights.index(min(adaptive_weights))
                hm_tea = hm_teas[best_teacher_idx]
                target_dicts = target_dict_teas[best_teacher_idx]

                kd_hm_loss_raw = self.get_sample_kd_loss_hm(hm_tea, hm_stu, loss_cfg, target_dicts, idx)
                kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw
                kd_reg_loss += loss_cfg.REG_LOSS.weight * self.get_sample_kd_loss_reg(
                    loss_cfg, cur_pred_teas[best_teacher_idx], cur_pred_stu, target_dicts, idx
                )

            elif softhm == 'addition':
                # Average losses from all teachers
                kd_hm_loss_raw = sum(self.get_sample_kd_loss_hm(hm_tea, hm_stu, loss_cfg, target_dict_tea, idx)
                                     for hm_tea, target_dict_tea in zip(hm_teas, target_dict_teas)) / 5
                kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw

                kd_reg_loss_raw = sum(
                    self.get_sample_kd_loss_reg(loss_cfg, cur_pred_tea, cur_pred_stu, target_dict_tea, idx)
                    for cur_pred_tea, target_dict_tea in zip(cur_pred_teas, target_dict_teas)) / 5
                kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw

            elif softhm == 'Soft-lr':
                # Weighted average based on adaptive weights  (1-e)get_confidence_weights (e-z)normalized
                normalized_weights = self.get_normalized_weights(adaptive_weights, temperature=temperature, adaptive_mask=None)#lr-softmax
                # normalized_weights = self.get_confidence_weights(adaptive_weights, temperature=temperature,
                #                                                  adaptive_mask=adaptive_mask)  # lr-softmax

                kd_hm_loss_raw = sum(w * self.get_sample_kd_loss_hm(hm_tea, hm_stu, loss_cfg, target_dict_tea, idx)
                                     for w, hm_tea, target_dict_tea in
                                     zip(normalized_weights, hm_teas, target_dict_teas))
                kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw

                kd_reg_loss_raw = sum(
                    w * self.get_sample_kd_loss_reg(loss_cfg, cur_pred_tea, cur_pred_stu, target_dict_tea, idx)
                    for w, cur_pred_tea, target_dict_tea in zip(normalized_weights, cur_pred_teas, target_dict_teas))
                kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw
            elif softhm == 'Soft':
                # Weighted average based on adaptive weights
                normalized_weights = self.get_normalized_weights(adaptive_weights, temperature=temperature, adaptive_mask=adaptive_mask)

                kd_hm_loss_raw = sum(w * self.get_sample_kd_loss_hm(hm_tea, hm_stu, loss_cfg, target_dict_tea, idx)
                                     for w, hm_tea, target_dict_tea in
                                     zip(normalized_weights, hm_teas, target_dict_teas))
                kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw

                kd_reg_loss_raw = sum(
                    w * self.get_sample_kd_loss_reg(loss_cfg, cur_pred_tea, cur_pred_stu, target_dict_tea, idx)
                    for w, cur_pred_tea, target_dict_tea in zip(normalized_weights, cur_pred_teas, target_dict_teas))
                kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw
            elif softhm == 'Hard':
                teacher_accuracies = self.model_cfg.LOGIT_KD.TEA_ACCUR
                total_acc = sum(teacher_accuracies)
                normalized_weights = [acc / total_acc for acc in teacher_accuracies]
                kd_hm_loss_raw = sum(w * self.get_sample_kd_loss_hm(hm_tea, hm_stu, loss_cfg, target_dict_tea, idx)
                                     for w, hm_tea, target_dict_tea in
                                     zip(normalized_weights, hm_teas, target_dict_teas))
                kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw

                kd_reg_loss_raw = sum(
                    w * self.get_sample_kd_loss_reg(loss_cfg, cur_pred_tea, cur_pred_stu, target_dict_tea, idx)
                    for w, cur_pred_tea, target_dict_tea in zip(normalized_weights, cur_pred_teas, target_dict_teas))
                kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw

            if self.kd_hm_sort_loss_func is not None and loss_cfg.HM_SORT_LOSS.weight != 0:
                kd_hm_sort_loss = self.kd_hm_sort_loss_func(hm_stu, hm_tea)
                kd_hm_sort_loss = loss_cfg.HM_SORT_LOSS.weight * kd_hm_sort_loss
            else:
                kd_hm_sort_loss = 0.0
        kd_loss = (kd_hm_loss + kd_hm_sort_loss + kd_reg_loss) / len(pred_stu)
        return kd_loss, kd_hm_loss / len(pred_stu), kd_reg_loss / len(pred_stu), kd_hm_sort_loss / len(pred_stu), adaptive_weights

    def get_kd_loss_with_target_tea(self, pred_tea, loss_cfg, target_dict_tea):
        """
        Args:
            pred_tea: pred_dict of teacher
                center: [bs, 2, feat_h, feat_w]. Offset to the nearest center
                center_z: [bs, 1, feat_h, feat_w]. absolute coordinates


            loss_cfg: kd loss config
            target_dict_tea

        Returns:

        """
        pred_stu = self.dense_head.forward_ret_dict['pred_dicts']

        kd_hm_loss = 0
        kd_reg_loss = 0

        for idx, cur_pred_stu in enumerate(pred_stu):
            cur_pred_tea = pred_tea[idx]
            target_hm = target_dict_tea['heatmaps'][idx]

            # interpolate if needed
            if (target_hm.shape != cur_pred_stu['hm'].shape) and loss_cfg.HM_LOSS.weight != 0 and \
                    self.model_cfg.LOGIT_KD.get('ALIGN', None):
                hm_tea, hm_stu = self.align_feature_map(
                    target_hm, cur_pred_stu['hm'], self.model_cfg.LOGIT_KD.ALIGN
                )
            else:
                hm_tea, hm_stu = target_hm, cur_pred_stu['hm']

            # classification loss
            if loss_cfg.HM_LOSS.weight == 0:
                kd_hm_loss_raw = 0
            elif loss_cfg.HM_LOSS.type == 'FocalLossCenterNet':
                kd_hm_loss_raw = self.kd_hm_loss_func(hm_stu, hm_tea)
            elif loss_cfg.HM_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
                kd_hm_loss_all = self.kd_hm_loss_func(hm_stu, hm_tea)
                # position-wise confidence mask: shape [bs, c, h, w]
                mask = (torch.max(hm_tea, dim=1)[0] > loss_cfg.HM_LOSS.thresh).float()
                if loss_cfg.HM_LOSS.get('fg_mask', None):
                    fg_mask = self.cal_fg_mask_from_target_hm(
                        target_dict_tea['heatmaps'][idx], hm_stu.shape
                    ).squeeze(1)
                    mask *= fg_mask
                kd_hm_loss_raw = (kd_hm_loss_all * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-6)
            else:
                raise NotImplementedError
            kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw

            # localization loss
            pred_boxes_stu = torch.cat([cur_pred_stu[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)
            if loss_cfg.REG_LOSS.weight == 0:
                kd_reg_loss_raw = 0
            elif loss_cfg.REG_LOSS.type == 'RegLossCenterNet':
                # parse teacher prediction to target style
                pred_boxes_tea = torch.cat([cur_pred_tea[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)

                # interpolate if the shape of feature map not match
                if (pred_boxes_tea.shape != pred_boxes_stu.shape) and self.model_cfg.LOGIT_KD.get('ALIGN', None):
                    pred_boxes_tea, pred_boxes_stu = self.align_feature_map(
                        pred_boxes_tea, pred_boxes_stu, self.model_cfg.LOGIT_KD.ALIGN
                    )

                pred_boxes_tea_selected = loss_utils._transpose_and_gather_feat(pred_boxes_tea,
                                                                                target_dict_tea['inds'][idx])

                kd_reg_loss_raw = self.reg_loss_func(
                    pred_boxes_stu, target_dict_tea['masks'][idx], target_dict_tea['inds'][idx], pred_boxes_tea_selected
                )
                kd_reg_loss_raw = (kd_reg_loss_raw * kd_reg_loss_raw.new_tensor(
                    loss_cfg.REG_LOSS.code_weights)).sum()
            else:
                raise NotImplementedError
            kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw

        kd_loss = (kd_hm_loss + kd_reg_loss) / len(pred_stu)

        return kd_loss, kd_hm_loss / len(pred_stu), kd_reg_loss / len(pred_stu)

    def get_kd_loss_with_decoded_boxes(self, pred_tea, loss_cfg, dense_head):
        """
        Args:
            pred_tea: list. [batch_size]
                pred_scores:
                pred_boxes:
                pred_labels
            loss_cfg:

        Returns:

        """
        pred_stu = dense_head.forward_ret_dict['decoded_pred_dicts']
        batch_kd_hm_loss = 0
        batch_kd_reg_loss = 0
        for b_idx, cur_pred_stu in enumerate(pred_stu):
            cur_pred_tea = pred_tea[b_idx]
            # filter boxes by confidence with a given threshold
            score_idx_stu = (cur_pred_stu['pred_scores'] >= loss_cfg.PRED_FILTER.score_thresh).nonzero().squeeze(-1)
            score_idx_tea = (cur_pred_tea['pred_scores'] >= loss_cfg.PRED_FILTER.score_thresh).nonzero().squeeze(-1)

            # filter boxes by iou
            iou_mask_stu, iou_mask_tea = self.filter_boxes_by_iou(
                cur_pred_stu['pred_boxes'][score_idx_stu], cur_pred_tea['pred_boxes'][score_idx_tea], loss_cfg
            )

            valid_idx_stu = score_idx_stu[iou_mask_stu]
            valid_idx_tea = score_idx_tea[iou_mask_tea]

            if valid_idx_stu.shape[0] == 0 or valid_idx_tea.shape[0] == 0:
                continue

            # confidence loss
            if loss_cfg.HM_LOSS.type == 'WeightedSmoothL1Loss':
                kd_hm_loss_all = self.kd_hm_loss_func(
                    cur_pred_stu['pred_scores'][None, valid_idx_stu, None],
                    cur_pred_tea['pred_scores'][None, valid_idx_tea, None].detach()
                )
                batch_kd_hm_loss += kd_hm_loss_all.mean()
            else:
                raise NotImplementedError

            # box regression loss
            if loss_cfg.REG_LOSS.type == 'WeightedSmoothL1Loss':
                valid_boxes_stu, valid_boxes_tea = self.add_sin_difference(
                    cur_pred_stu['pred_boxes'][valid_idx_stu], cur_pred_tea['pred_boxes'][valid_idx_tea]
                )

                kd_reg_loss_all = self.kd_reg_loss_func(
                    valid_boxes_stu.unsqueeze(0), valid_boxes_tea.unsqueeze(0).detach()
                )
                batch_kd_reg_loss += kd_reg_loss_all.mean()
            else:
                raise NotImplementedError

        kd_hm_loss = batch_kd_hm_loss * loss_cfg.HM_LOSS.weight / len(pred_stu)
        kd_reg_loss = batch_kd_reg_loss * loss_cfg.REG_LOSS.weight / len(pred_stu)

        kd_loss = kd_hm_loss + kd_reg_loss

        return kd_loss, kd_hm_loss, kd_reg_loss
