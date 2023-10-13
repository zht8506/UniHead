# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_conv_layer, bias_init_with_prob, constant_init, is_norm,normal_init
from .DAT_Transformer import DAT
from .CIT_Transformer import CIT,PosCNN
from timm.models.layers import trunc_normal_
from ..builder import HEADS
from .anchor_head import AnchorHead
import torch


@HEADS.register_module()
class retina_Uni(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = retina_Uni(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(retina_Uni, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        dcn_cfg=dict(type='DCNv2', deform_groups=1)
        self.stem = build_conv_layer(
            dcn_cfg,
            self.in_channels,
            self.feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False)

        self.Csw_block1 = DAT(self.feat_channels,16 ,1)
        self.Csw_block2 = DAT(self.feat_channels, 16, 1)

        self.pos_ca_cls=PosCNN(self.feat_channels,self.feat_channels)
        self.pos_ca_reg=PosCNN(self.feat_channels,self.feat_channels)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_convs_2 = nn.ModuleList()
        self.reg_convs_2 = nn.ModuleList()

        self.cls_convs.append(
            CIT(self.feat_channels, 2, mlp_ratio=2, reducedim=False,out=True)
        )
        self.cls_convs_2.append(
            CIT(self.feat_channels, 2, mlp_ratio=2, reducedim=False,out=True)
        )
        self.reg_convs.append(
            CIT(self.feat_channels, 2, mlp_ratio=2, reducedim=False,out=True)
        )
        self.reg_convs_2.append(
            CIT(self.feat_channels, 2, mlp_ratio=2, reducedim=False,out=True)
        )

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

            ####add by davina
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # Use prior in model initialization to improve stability
        bias_cls = bias_init_with_prob(0.01)
        torch.nn.init.constant_(self.retina_cls.bias, bias_cls)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        x=self.stem(x)
        decouple_x=self.Csw_block1(x)
        decouple_x = self.Csw_block2(decouple_x)
        cls_feat, reg_feat=decouple_x,decouple_x
        cls_feat=self.pos_ca_cls(cls_feat)
        reg_feat=self.pos_ca_reg(reg_feat)
        cls_feat_=None
        reg_feat_=None


        for cls_conv in self.cls_convs:
            cls_feat_ = cls_conv(cls_feat,reg_feat)
        for reg_conv in self.reg_convs:
            reg_feat_ = reg_conv(reg_feat,cls_feat)

        for cls_conv in self.cls_convs_2:
            cls_feat = cls_conv(cls_feat_,reg_feat_)
        for reg_conv in self.reg_convs_2:
            reg_feat = reg_conv(reg_feat_,cls_feat_)

        # for cls_conv in self.cls_convs_3:
        #     cls_feat_ = cls_conv(cls_feat,reg_feat)
        # for reg_conv in self.reg_convs_3:
        #     reg_feat_ = reg_conv(reg_feat,cls_feat)

        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred