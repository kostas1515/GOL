_base_ = [
    '../lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis_v1.py'
]

# data = dict(train=dict(oversample_thr=0.0),samples_per_gpu=1)

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-6.5, override=dict(name='fc_cls')))))

work_dir='./experiments/r101_rfs_4x4_2x_softmax/'
# work_dir='./experiments/test/'