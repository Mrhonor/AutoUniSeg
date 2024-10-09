_base_ = [
    '../_base_/models/afformer.py', '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
afformer_model = dict(
    pretrained='./pretained_weight/AFFormer_tiny_ImageNet1k.pth',
    backbone=dict(
        type='afformer_tiny',
        strides=[4, 2, 2, 2]),
    decode_head=dict(
        type='CLS',
        in_channels=[216],
        in_index=[3],
        channels=256,
        aff_channels=256,
        aff_kwargs=dict(MD_R=16),
        num_classes=256,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        dropout_ratio=0.1,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

