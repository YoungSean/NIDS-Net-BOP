import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import logging
import numpy as np
from src.utils.bbox_utils import CropResizePad, CustomResizeLongestSide
from torchvision.utils import make_grid, save_image
from src.model.utils import BatchedData
from copy import deepcopy

descriptor_size = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14": 1536,
}


def crop_images_and_masks(images, masks, bboxes, img_size=224):
    """
    Crop images and masks according to the given bounding boxes.

    Parameters:
        images (torch.Tensor): Tensor of shape [batch_size, channels, height, width].
        masks (torch.Tensor): Tensor of shape [batch_size, height, width].
        bboxes (torch.Tensor): Tensor of shape [batch_size, 4] with each row [y1, x1, y2, x2].

    Returns:
        cropped_images (torch.Tensor): Tensor of cropped images.
        cropped_masks (torch.Tensor): Tensor of cropped masks.
    """
    cropped_images = []
    cropped_masks = []

    for image, mask, bbox in zip(images, masks, bboxes):
        x0, y0, x1, y1 = bbox
        cropped_image = image[:, y0:y1, x0:x1]
        cropped_mask = mask[:, y0:y1, x0:x1]
        cropped_image = F.interpolate(cropped_image.unsqueeze(0), size=(img_size, img_size), mode='bicubic')
        cropped_mask = F.interpolate(cropped_mask.unsqueeze(0), size=(img_size, img_size), mode='bicubic')

        cropped_images.append(cropped_image)
        cropped_masks.append(cropped_mask)

    # Stack the list of tensors into a single tensor
    cropped_images = torch.cat(cropped_images, dim=0)
    cropped_masks = torch.cat(cropped_masks, dim=0)

    return cropped_images, cropped_masks

class CustomDINOv2(pl.LightningModule):
    def __init__(
        self,
        model_name,
        model,
        token_name,
        image_size,
        chunk_size,
        descriptor_width_size,
        patch_size=14,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.token_name = token_name
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size
        logging.info(f"Init CustomDINOv2 done!")
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.img_resize = T.Compose(
            [
                T.Resize((patch_size, patch_size), interpolation=T.InterpolationMode.BICUBIC)
            ])
        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        logging.info(
            f"Init CustomDINOv2 with full size={descriptor_width_size} and proposal size={self.proposal_size} done!"
        )

    def process_rgb_proposals(self, image_np, masks, boxes):
        """
        1. Normalize image with DINOv2 transfom
        2. Mask and crop each proposals
        3. Resize each proposals to predefined longest image size
        """
        num_proposals = len(masks)
        masks = masks.unsqueeze(1)
        rgb = self.rgb_normalize(image_np).to(masks.device).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        cropped_imgs, cropped_masks = crop_images_and_masks(rgbs, masks, boxes, self.proposal_size)
        # masked_rgbs = rgbs * masks.unsqueeze(1)
        # processed_masked_rgbs = self.rgb_proposal_processor(
        #     rgbs, boxes
        # )  # [N, 3, target_size, target_size]
        # processed_masks = self.rgb_proposal_processor(
        #     masks, boxes
        # )  # [N, 1, target_size, target_size]
        # return processed_masked_rgbs, processed_masks
        return cropped_imgs, cropped_masks
    @torch.no_grad()
    def compute_features(self, images, token_name, masks):
        if token_name == "x_norm_clstoken":
            if images.shape[0] > self.chunk_size:
                features = self.forward_by_chunk(images, masks)
            else:
                # features = self.model(images)
                emb = self.model.forward_features(images)
                grid = emb["x_norm_patchtokens"].detach()
                # masks = masks.unsqueeze(1)
                mask_size = masks.size(2) // 14
                grid = grid.view(len(images), mask_size, mask_size, -1)
                masks = F.interpolate(masks, size=(mask_size, mask_size), mode='bilinear')
                features = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(
                    -1)

        else:  # get both features
            raise NotImplementedError
        return features

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs, processed_masks):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        batch_masks = BatchedData(batch_size=self.chunk_size, data=processed_masks)
        del processed_rgbs  # free memory
        del processed_masks
        features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_features(
                batch_rgbs[idx_batch], token_name="x_norm_clstoken", masks=batch_masks[idx_batch]
            )
            features.cat(feats)
        return features.data


    @torch.no_grad()
    def forward_cls_token(self, image_np, proposals):
        processed_rgbs, processed_masks = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        return self.forward_by_chunk(processed_rgbs, processed_masks)

    @torch.no_grad()
    def forward(self, image_np, proposals):
        return self.forward_cls_token(image_np, proposals)

