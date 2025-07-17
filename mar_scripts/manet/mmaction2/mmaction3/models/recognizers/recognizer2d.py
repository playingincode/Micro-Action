# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels,embs_la, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        batches = imgs.shape[0]
        # print("Images shape",imgs.shape)
        imgs=imgs.squeeze(2)
        imgs=imgs.permute(0,2,1)
        # print("After Permute and squeeze shape",imgs.shape)

        mask_bool = torch.ones((batches, 10), dtype=torch.bool)
        mask_bool = mask_bool.unsqueeze(1).cuda()

        # print("Images shape",imgs.shape)
        # print("Mask book",mask_bool.shape)

        # imgs_1,_= self.SGP_block(imgs,mask_bool)

        #What Aglind did
        # imgs_2, _ = self.SGP_block_2(imgs, mask_bool)
        # # print(imgs_1.shape, imgs_2.shape)

        # imgs_1 = imgs_1.permute(2, 0, 1)  # [T1, B, C] - query
        # imgs_2 = imgs_2.permute(2, 0, 1)  # [T2, B, C] - key & value

        # attn_output, _ = self.attn(query=imgs_1, key=imgs_2, value=imgs_2)
        # imgs = attn_output.permute(0, 2, 1)
        
        # imgs = imgs + self.Global_Relational_Block(self.norm_layer(imgs))
        # print("Images shape",imgs.shape)
        p=self.Global_Relational_Block(imgs.permute(0,2,1))
        # print("P shape",p.shape)
        imgs = imgs + p.permute(0,2,1)
        y,_=self.SGP_block(imgs,mask_bool)
        imgs = imgs + y
        imgs = imgs.permute(0, 2, 1)




        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()
        # print("Images shape",imgs.shape)
        # exit()
        # x = self.extract_feat(imgs)
        # print("X shape",x.shape)
        x=imgs

        if self.backbone_from in ['torchvision', 'timm']:
            # print("Backbone from",self.backbone_from)
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if self.with_neck:
            # print("Neck",self.with_neck)
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x, labels.squeeze())
            x = x.squeeze(2)
            num_segs = 1
            losses.update(loss_aux)
        # print("X shape",x.shape)
        # x = x.squeeze(1)
        cls_score,emb_score = self.cls_head(x, num_segs)#8,59   8,300
        gt_labels = labels.squeeze()#8
        # print("Classifier score",cls_score.shape)
        # print("Embedding",emb_score.shape)
        # print("gt_labels",gt_labels.shape)
        # print("embs_la",embs_la.shape)

        loss_cls = self.cls_head.loss(cls_score, emb_score,gt_labels,embs_la, **kwargs)#在base的loss里面
        losses.update(loss_cls)


        return losses

    def _do_test(self, imgs,labels,embs_la):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs=imgs.squeeze(2)
        imgs=imgs.permute(0,2,1)
        mask_bool = torch.ones((batches, 10), dtype=torch.bool)
        mask_bool = mask_bool.unsqueeze(1).cuda()
        # print("Images shape",imgs.shape)
        # print("Mask book",mask_bool.shape)
        # imgs_1, _ = self.SGP_block(imgs, mask_bool)

        # imgs_2, _ = self.SGP_block_2(imgs, mask_bool)
        p=self.Global_Relational_Block(imgs.permute(0,2,1))
        # print("P shape",p.shape)
        imgs = imgs + p.permute(0,2,1)
        y,_=self.SGP_block(imgs,mask_bool)
        imgs = imgs + y
        imgs = imgs.permute(0, 2, 1)
        # print(imgs_1.shape, imgs_2.shape)

        # imgs_1 = imgs_1.permute(2, 0, 1)  # [T1, B, C] - query
        # imgs_2 = imgs_2.permute(2, 0, 1)  # [T2, B, C] - key & value

        # attn_output, _ = self.attn(query=imgs_1, key=imgs_2, value=imgs_2)
        # imgs = attn_output.permute(0, 2, 1)
        # imgs = imgs.permute(0, 2, 1)
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        # x = self.extract_feat(imgs)
        x=imgs

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))


        if self.with_neck:
            print(self.with_neck)
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        # if self.feature_extraction:
        #     # perform spatial pooling
        #     avg_pool = nn.AdaptiveAvgPool2d(1)
        #     x = avg_pool(x)
        #     # squeeze dimensions
        #     x = x.reshape((batches, num_segs, -1))
        #     # temporal average pooling
        #     x = x.mean(axis=1)
        #     return x

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`

        # should have cls_head if not extracting features
        # x = x.squeeze(1)
        cls_score,_ = self.cls_head(x, num_segs)#8,59

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)

        return cls_score

    def _do_fcn_test(self, imgs):
        # [N, num_crops * num_segs, C, H, W] ->
        # [N * num_crops * num_segs, C, H, W]
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = self.test_cfg.get('num_segs', self.backbone.num_segments)

        if self.test_cfg.get('flip', False):
            imgs = torch.flip(imgs, [-1])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
        else:
            x = x.reshape((-1, num_segs) +
                          x.shape[1:]).transpose(1, 2).contiguous()

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        cls_score = self.cls_head(x, fcn_test=True)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

    def forward_test(self, imgs,labels,embs_la):
        """Defines the computation performed at every call when evaluation and
        testing."""
        if self.test_cfg.get('fcn_test', False):
            # If specified, spatially fully-convolutional testing is performed
            assert not self.feature_extraction
            assert self.with_cls_head
            return self._do_fcn_test(imgs).cpu().numpy()
        return self._do_test(imgs,labels,embs_la).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        outs = self.cls_head(x, num_segs)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
