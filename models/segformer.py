import torch
from transformers import SegformerForSemanticSegmentation

def get_segformer_b5(num_classes=2, pretrained_model="nvidia/segformer-b5-finetuned-ade-640-640"):
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    return model
