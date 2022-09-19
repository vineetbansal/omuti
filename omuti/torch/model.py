# Model
import torchvision
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.retinanet import AnchorGenerator


def create_anchor_generator(sizes=((8, 16, 32, 64, 128, 256, 400),),
                            aspect_ratios=((0.5, 1.0, 2.0),)):
    """
    Create anchor box generator as a function of sizes and aspect ratios
    Documented https://github.com/pytorch/vision/blob/67b25288ca202d027e8b06e17111f1bcebd2046c/torchvision/models/detection/anchor_utils.py#L9
    let's make the network generate 5 x 3 anchors per spatial
    location, with 5 different sizes and 3 different aspect
    ratios. We have a Tuple[Tuple[int]] because each feature
    map could potentially have different sizes and
    aspect ratios
    Args:
        sizes:
        aspect_ratios:

    Returns: anchor_generator, a pytorch module

    """
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

    return anchor_generator


def create_model(num_classes, nms_thresh, score_thresh):
    resnet = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

    model = RetinaNet(backbone=resnet.backbone, num_classes=num_classes)
    model.nms_thresh = nms_thresh
    model.score_thresh = score_thresh

    return model
