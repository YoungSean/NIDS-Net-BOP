import torch
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
import logging
import os
import os.path as osp
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from src.utils.inout import save_json, load_json, save_json_bop23
from src.model.utils import BatchedData, Detections, convert_npz_to_json
from hydra.utils import instantiate
import time
import glob
from functools import partial
import multiprocessing
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
import json
from torch import nn

def stableMatching(preferenceMat):
    """
    Compute Stable Matching
    """
    mDict = dict()

    engageMatrix = np.zeros_like(preferenceMat)
    for i in range(preferenceMat.shape[0]):
        tmp = preferenceMat[i]
        sortIndices = np.argsort(tmp)[::-1]
        mDict[i] = sortIndices.tolist()

    freeManList = list(range(preferenceMat.shape[0]))

    while freeManList:
        curMan = freeManList.pop(0)
        curWoman = mDict[curMan].pop(0)
        if engageMatrix[:, curWoman].sum() == 0:
            engageMatrix[curMan, curWoman] = 1
        else:
            engagedMan = np.where(engageMatrix[:, curWoman] == 1)[0][0]
            if preferenceMat[engagedMan, curWoman] > preferenceMat[curMan, curWoman]:
                freeManList.append(curMan)
            else:
                engageMatrix[engagedMan, curWoman] = 0
                engageMatrix[curMan, curWoman] = 1
                freeManList.append(engagedMan)
    return engageMatrix

class CNOS(pl.LightningModule):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model

        self.onboarding_config = onboarding_config
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config
        self.log_interval = log_interval
        self.log_dir = log_dir

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)
        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )
        logging.info(f"Init CNOS done!")
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor(vit_model="vit_l")
        logging.info("Initialize GDINO and SAM done!")

    def set_reference_objects(self):
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data = {"descriptors": BatchedData(None)}
        descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors.pth")
        vox_descriptors_path = osp.join(self.ref_dataset.template_dir, "lmo_object_features.json")
        if self.onboarding_config.rendering_type == "pbr":
            descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["descriptors"] = torch.load(descriptors_path).to(self.device)
            # with open(os.path.join(vox_descriptors_path), 'r') as f:
            #     feat_dict = json.load(f)
            #
            # object_features = torch.Tensor(feat_dict['features']).cuda()
            # self.ref_data["descriptors"] = object_features.view(8, -1, 1024)
            # print("using my vox features")
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
                ref_feats = self.descriptor_model.compute_features(
                    ref_imgs, token_name="x_norm_clstoken", masks=ref_masks
                )
                self.ref_data["descriptors"].append(ref_feats)

            self.ref_data["descriptors"].stack()  # N_objects x descriptor_size
            self.ref_data["descriptors"] = self.ref_data["descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["descriptors"], descriptors_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Descriptors shape: {self.ref_data['descriptors'].shape}"
        )

    def move_to_device(self):
        self.descriptor_model.model = self.descriptor_model.model.to(self.device)
        self.descriptor_model.model.device = self.device
        # if there is predictor in the model, move it to device
        if hasattr(self.segmentor_model, "predictor"):
            self.segmentor_model.predictor.model = (
                self.segmentor_model.predictor.model.to(self.device)
            )
        else:
            self.segmentor_model.model.setup_model(device=self.device, verbose=True)
        logging.info(f"Moving models to {self.device} done!")

    def find_matched_proposals(self, proposal_decriptors):
        # compute matching scores for each proposals
        scores = self.matching_config.metric(
            proposal_decriptors, self.ref_data["descriptors"]
        )  # N_proposals x N_objects x N_templates
        if self.matching_config.aggregation_function == "mean":
            score_per_proposal_and_object = (
                torch.sum(scores, dim=-1) / scores.shape[-1]
            )  # N_proposals x N_objects
        elif self.matching_config.aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "avg_5":
            score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
        else:
            raise NotImplementedError

        # assign each proposal to the object with highest scores
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_query
        do_matching = False
        if do_matching:
            score_per_proposal = []
            assigned_idx_object = []
            sims = score_per_proposal_and_object
            num_object = sims.shape[1]
            num_proposals = sims.shape[0]
            # ------------ ranking and sorting ------------
            # Initialization
            sel_obj_ids = [str(v) for v in list(np.arange(num_object))]  # ids for selected obj
            sel_roi_ids = [str(v) for v in list(np.arange(num_proposals))]  # ids for selected roi

            # Padding
            max_len = max(len(sel_roi_ids), len(sel_obj_ids))
            sel_sims_symmetric = torch.ones((max_len, max_len)) * -1
            sel_sims_symmetric[:len(sel_roi_ids), :len(sel_obj_ids)] = sims.clone()

            pad_len = abs(len(sel_roi_ids) - len(sel_obj_ids))
            if len(sel_roi_ids) > len(sel_obj_ids):
                pad_obj_ids = [str(i) for i in range(num_object, num_object + pad_len)]
                sel_obj_ids += pad_obj_ids
            elif len(sel_roi_ids) < len(sel_obj_ids):
                pad_roi_ids = [str(i) for i in range(len(sel_roi_ids), len(sel_roi_ids) + pad_len)]
                sel_roi_ids += pad_roi_ids

            # ------------ stable matching ------------
            matchedMat = stableMatching(
                sel_sims_symmetric.detach().data.cpu().numpy())  # predMat is raw predMat
            predMat_row = np.zeros_like(
                sel_sims_symmetric.detach().data.cpu().numpy())  # predMat_row is the result after stable matching
            Matches = dict()
            for i in range(matchedMat.shape[0]):
                tmp = matchedMat[i, :]
                a = tmp.argmax()
                predMat_row[i, a] = tmp[a]
                Matches[sel_roi_ids[i]] = sel_obj_ids[int(a)]

            # ------------ thresholding ------------
            preds = Matches.copy()
            # ------------ save per scene results ------------
            for k, v in preds.items():
                if int(k) >= num_proposals:  # since the number of proposals is less than the number of object features
                    break

                if int(v) >= num_object:
                    score_per_proposal.append(0.0) #will ignore this proposal later
                    assigned_idx_object.append(-1)
                    continue

                # if float(sims[int(k), int(v)]) < score_thresh_predefined:
                #     continue

                score_per_proposal.append(float(sims[int(k), int(v)]))
                assigned_idx_object.append(int(v))
            score_per_proposal = torch.tensor(score_per_proposal).to(proposal_decriptors.device)
            assigned_idx_object = torch.tensor(assigned_idx_object).to(proposal_decriptors.device)
        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        pred_scores = score_per_proposal[idx_selected_proposals]
        return idx_selected_proposals, pred_idx_objects, pred_scores

    def test_step(self, batch, idx):
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}",
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0])
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        image_np = np.uint8(image_np.clip(0, 1) * 255)

        # run propoals
        proposal_stage_start_time = time.time()
        image_pil = Image.fromarray(image_np).convert("RGB")
        bboxes, phrases, gdino_conf = self.gdino.predict(image_pil, "objects")
        w, h = image_pil.size  # Get image width and height
        # Scale bounding boxes to match the original image size
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
        image_pil_bboxes, masks = self.SAM.predict(image_pil, image_pil_bboxes)
        proposals = dict()
        proposals["masks"] = masks.squeeze(1).to(torch.float32)  # to N x H x W, torch.float32 type as the output of fastSAM
        proposals["boxes"] = image_pil_bboxes

        # proposals = self.segmentor_model.generate_masks(image_np)

        # init detections with masks and boxes
        detections = Detections(proposals)
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )
        # compute descriptors
        query_decriptors = self.descriptor_model(image_np, detections)
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            pred_scores,
        ) = self.find_matched_proposals(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        detections.add_attribute("scores", pred_scores)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0

    def test_epoch_end(self, outputs):
        if self.global_rank == 0:  # only rank 0 process
            # can use self.all_gather to gather results from all processes
            # but it is simpler just load the results from files so no file is missing
            result_paths = sorted(
                glob.glob(
                    osp.join(
                        self.log_dir,
                        f"predictions/{self.dataset_name}/{self.name_prediction_file}/*.npz",
                    )
                )
            )
            result_paths = sorted(
                [path for path in result_paths if "runtime" not in path]
            )
            num_workers = 10
            logging.info(f"Converting npz to json requires {num_workers} workers ...")
            pool = multiprocessing.Pool(processes=num_workers)
            convert_npz_to_json_with_idx = partial(
                convert_npz_to_json,
                list_npz_paths=result_paths,
            )
            detections = list(
                tqdm(
                    pool.imap_unordered(
                        convert_npz_to_json_with_idx, range(len(result_paths))
                    ),
                    total=len(result_paths),
                    desc="Converting npz to json",
                )
            )
            formatted_detections = []
            for detection in tqdm(detections, desc="Loading results ..."):
                formatted_detections.extend(detection)

            detections_path = f"{self.log_dir}/{self.name_prediction_file}.json"
            save_json_bop23(detections_path, formatted_detections)
            logging.info(f"Saved predictions to {detections_path}")
