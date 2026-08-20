"""
Model builders for the two recommended benchmark models: RetinaNet and FCOS.

Both come from torchvision.models.detection and both:
  - accept the same input format: list of image tensors (0-1 float, CHW) +
    list of target dicts with 'boxes' (xyxy absolute) and 'labels' (int64, 0=background)
  - return a dict of losses in train() mode, and a list of prediction dicts in eval() mode
  - handle their own internal resize/normalize (image_mean/image_std below),
    so make sure you're using transforms_torchvision.py (no A.Normalize()) upstream.

Both default to a ResNet50+FPN backbone, which is NOT lightweight - this is
fine for benchmarking (these are reference/ceiling models per the earlier
discussion), but not what you'd deploy onboard. A note on lighter backbones
is included at the bottom if you want to push either toward something more
deployment-realistic later.
"""

import torchvision
import torch.nn as nn
from functools import partial
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import (
    fcos_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights,
)
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, RetinaNetClassificationHead
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

# ImageNet stats - since transforms_torchvision.py only scales to 0-1 and does NOT
# apply mean/std normalization, we let the model's internal transform do that here.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Custom anchor sizes, derived directly from box_stats.py output:
#
#   PERSON  effective_size: p5=6.3  median=17.4  p95=49.2   (h/w: p5=0.86 median=1.95 p95=3.27)
#   VEHICLE effective_size: p5=9.4  median=31.9  p95=113.3  (h/w: p5=0.39 median=0.86 p95=2.05)
#
# torchvision's default RetinaNet anchors start at 32px (smallest FPN level)
# up to ~812px - person's ENTIRE distribution sits mostly below that floor.
# The sizes below shift the smallest level down to 8-16px so person boxes
# actually have anchors close to their true scale, while still keeping
# coverage up through vehicle's large tail (max ~983px, via the 384-512
# level below).
#
# Aspect ratios widened from the default (0.5, 1.0, 2.0) to also cover
# person's tall tail (p95=3.27) and vehicle's wide tail (p5=0.39).
# ---------------------------------------------------------------------------
LONG_SIDE = 960 

CUSTOM_ANCHOR_SIZES = (
    (4, 6, 8), # P2 stride 4 evven smaller
    (8, 11, 16),        # P3, stride 8  - tiny pedestrians (person p5-p25 range)
    (24, 32, 40),       # P4, stride 16 - typical pedestrians + small vehicles (person median-p95, vehicle p5-p25)
    (48, 64, 96),       # P5, stride 32 - typical vehicles (vehicle median-p75)
    (128, 192, 256),    # P6, stride 64 - large vehicles (vehicle p95 range)
    (256, 384, 512),    # P7, stride 128 - largest vehicle tail (buses/trucks up to ~983px)
)
CUSTOM_ASPECT_RATIOS = (
    (0.5, 1.0, 1.5, 2.0, 3.5),      # P2
    (0.5, 1.0, 1.5, 2.0, 3.5),      # P3
    (0.33, 0.5, 1.0, 2.0, 3.0),     # P4
    (0.25, 0.33, 0.5, 1.0, 2.0),    # P5
    (0.25, 0.33, 0.5, 1.0, 2.0),    # P6
    (0.25, 0.33, 0.5, 1.0, 2.0),    # P7
)


def build_custom_anchor_generator():
    return AnchorGenerator(
        sizes=CUSTOM_ANCHOR_SIZES,
        aspect_ratios=CUSTOM_ASPECT_RATIOS
    )


def build_retinanet(num_classes, pretrained_backbone=True):
    """
    num_classes: include background, e.g. 3 for {background, person, vehicle}

    IMPORTANT: this does NOT use torchvision's retinanet_resnet50_fpn_v2()
    convenience wrapper. That wrapper always constructs its own anchor
    generator internally (_default_anchorgen()) and passes it as an
    explicit keyword to RetinaNet(...) - so a custom anchor_generator passed
    via **kwargs collides with it (TypeError: got multiple values for
    keyword argument 'anchor_generator'). There's no supported way to
    override this through the wrapper.

    Instead, this builds the model from the same lower-level pieces the
    wrapper itself uses (confirmed from torchvision's source): ResNet50
    backbone -> _resnet_fpn_extractor -> RetinaNetHead -> RetinaNet. This
    mirrors retinanet_resnet50_fpn_v2's construction exactly, except our
    anchor_generator (and the head sized to match it) replaces the default.

    Note: _resnet_fpn_extractor is a private (underscored) torchvision
    function - it's what the public wrapper calls internally, so it's the
    correct piece to use here, but as a private API it could change in a
    future torchvision release without notice. If this ever breaks after a
    torchvision upgrade, check torchvision.models.detection.retinanet's
    source for what replaced it.
    """
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
    backbone = resnet50(weights=weights_backbone, progress=True)
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_layers=3,  # matches retinanet_resnet50_fpn_v2's default when is_trained=True
        returned_layers=[1, 2, 3, 4],
        extra_blocks=LastLevelP6P7(2048, 256),
    )

    anchor_generator = build_custom_anchor_generator()

    head = RetinaNetHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        norm_layer=partial(nn.GroupNorm, 32),  # bias-only conv head, consistent with prior architecture across train/eval
    )
    head.regression_head._loss_type = "giou"  # matches the "v2" improvement (giou box loss instead of smooth L1)

    model = RetinaNet(
        backbone,
        num_classes,
        anchor_generator=anchor_generator,
        head=head,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        min_size=LONG_SIDE,
        max_size=LONG_SIDE,
        fg_iou_thresh=0.4,
        bg_iou_thresh=0.3
    )

    return model


def build_fcos(num_classes, pretrained_backbone=True):
    """
    num_classes: include background, e.g. 3 for {background, person, vehicle}
    """
    weights = FCOS_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
    model = fcos_resnet50_fpn(
        weights=weights,
        weights_backbone="DEFAULT" if pretrained_backbone else None,
        num_classes=None if weights else num_classes,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        min_size=640,
        max_size=640,
    )

    # same reasoning as build_retinanet above: always rebuild the head the
    # same way regardless of pretrained_backbone, so train-time and eval-time
    # architectures never diverge
    from torchvision.models.detection.fcos import FCOSClassificationHead
    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors  # FCOS uses 1 anchor/location but head expects this arg
    model.head.classification_head = FCOSClassificationHead(
        in_channels, num_anchors, num_classes,
        norm_layer=None,
    )

    return model


# ---------------------------------------------------------------------------
# OPTIONAL: pushing toward a lighter backbone later
# ---------------------------------------------------------------------------
# torchvision doesn't ship a MobileNet+RetinaNet or MobileNet+FCOS combo out of
# the box the way it does for Faster R-CNN (torchvision.models.detection.
# fasterrcnn_mobilenet_v3_large_fpn exists; retinanet/fcos don't have an
# equivalent convenience function). To get a lighter RetinaNet/FCOS you'd need
# to build a custom backbone + FeaturePyramidNetwork and pass it to the
# lower-level RetinaNet(...)/FCOS(...) constructors directly, e.g.:
#
#   from torchvision.models.detection.backbone_utils import BackboneWithFPN
#   from torchvision.models import mobilenet_v3_large
#   backbone = mobilenet_v3_large(weights="DEFAULT").features
#   # then wrap `backbone` in BackboneWithFPN with the right return_layers/
#   # in_channels_list for mobilenet's feature map stages.
#
# This is a reasonable follow-up once you know whether RetinaNet/FCOS's
# accuracy on person recall justifies the deployment engineering effort -
# no point building this until the ResNet50 benchmark tells you it's worth it.