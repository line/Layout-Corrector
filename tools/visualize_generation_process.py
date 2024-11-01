"""
Save Generation Process as pickle and images
"""
from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import os
import pickle
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from hydra.utils import instantiate
from trainer.corrector.util import CorrectorMaskingMode
from trainer.global_configs import DATASET_DIR
from trainer.helpers.sampling import SAMPLING_CONFIG_DICT
from trainer.helpers.util import load_config, set_seed
from trainer.helpers.visualization import save_image
from trainer.hydra_configs import TestConfig
from trainer.models.common.util import build_model, load_model
from trainer.helpers.metric import compute_alignment, compute_overlap

SIZE = (360, 240)
cond_type = "unconditional"
W_CANVAS = False  # # choices: unconditional, c, cwh, partial, refinement
RESULT_SAVE_TIMESTEPS = list(range(60, 100))


torch.backends.cudnn.deterministic = True


def run_generation(
    model: nn.Module,
    sampling_cfg: DictConfig | None = None,
    corrector: nn.Module | None = None,
    init_seed: int = 0,
    num_samples: int = 100
) -> tuple[list, list[dict], list]:
    """Run layout generation process with the given generator and corrector.
    
    Note:
        This method returns the intermediate results at each timestep.
    """
    set_seed(init_seed)

    # generate
    ids = model.model.sample(
        batch_size=num_samples, 
        cond=None, 
        sampling_cfg=sampling_cfg, 
        get_intermediate_results=True,  # get results at all timesteps
        corrector=corrector
    )
    # convert tensor to list to save as pickle
    ids_for_pkl = [_id.cpu().tolist() for _id in ids]
    decoded_preds = [model.tokenizer.decode(_id.cpu()) for _id in ids]
    decoded_preds_for_pkl = [
        dict(
            bbox=_pred["bbox"].cpu().long().tolist(),
            label=_pred["label"].cpu().tolist(),
            mask=_pred["mask"].cpu().tolist()
        )
        for _pred in decoded_preds
    ]
    return ids_for_pkl, decoded_preds_for_pkl, decoded_preds
    

def save_generation_process_as_image(
    pred_list: list[dict[str, torch.Tensor]],
    save_timesteps: list[int],
    corr_timesteps: list[int] | None = None,
    ncol: int = 10,
    out_dir: str = "./tmp",
    fname_prefix: str | None = None,
    **save_kwargs
) -> None:
    """Save layout image on each timestep, resulting in the visualizaiton of generation process
    
    Args:
        pred_list (list[dict[str, torch.Tensor]]): A list of predicted layouts on each timestep.
        save_timesteps (list[int]): A list of timesteps to be saved.
        ncol (int): The number of columns.
        out_dir (str): Output directory.
        fname_prefix (str | None): Output file name prefix.
        kwargs: including color, label names, canvas_size, and use_grid or not.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_timesteps = len(pred_list)
    save_timesteps.sort()
    assert num_timesteps > len(save_timesteps) and num_timesteps >= save_timesteps[-1]

    new_ncol = len(save_timesteps) if len(save_timesteps) < ncol else ncol

    B = pred_list[0]["bbox"].size(0)
    for b in range(B):
        nrow, row_mod = divmod(len(save_timesteps), new_ncol)
        if row_mod > 1:
            nrow += 1
        fig, axs = plt.subplots(nrow, new_ncol, figsize=(4 * new_ncol, 4 * nrow + 3) )
        for idx, t in enumerate(save_timesteps):
            _pred = pred_list[t]
            y, x = divmod(idx, new_ncol)
            result_image = save_image(
                _pred["bbox"][b].unsqueeze(0),   # (B, N, 4)
                _pred["label"][b].unsqueeze(0),  # (B, N)
                _pred["mask"][b].unsqueeze(0),   # (B, N)
                **save_kwargs
            )
            title_timestep = num_timesteps - t - 1
            title = fr"$t={title_timestep}$"
            if corr_timesteps is not None and title_timestep in corr_timesteps:            
                # use bold style
                title = r"$\mathbf{t=" + f"{title_timestep}" + r"}$"

            if len(save_timesteps) < ncol:
                axs[x].imshow(result_image)
                axs[x].set_title(title)
                axs[x].axis("off")
            else:
                axs[y, x].imshow(result_image)
                axs[y, x].set_title(title)
                axs[y, x].axis("off")
        plt.tight_layout()

        last_layout = pred_list[-1]
        alignments = compute_alignment(last_layout["bbox"][b].unsqueeze(0), last_layout["mask"][b].unsqueeze(0))
        alignment_score = alignments["alignment-LayoutGAN++"].sum().item() * 100
        overlaps = compute_overlap(last_layout["bbox"][b].unsqueeze(0), last_layout["mask"][b].unsqueeze(0))
        overlap_score = overlaps["overlap-LayoutGAN++"].sum().item()
        plt.suptitle(f"{fname_prefix} (Align: {alignment_score:.3f}, Overlap: {overlap_score:.3f})")

        fname = f"batch{b}_{save_timesteps[0]}to{num_timesteps}.jpg"
        if fname_prefix is not None:
            fname = f"{fname_prefix}_{fname}"

        plt.savefig(os.path.join(out_dir, fname))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_job_dir", 
                        type=str, 
                        help="Layout Generation Model's job_dir, where includes ckpt.")
    parser.add_argument("--corr_job_dir", 
                        type=str, 
                        help="LayoutCorrector's job_dir, where includes ckpt.", 
                        default=None)
    parser.add_argument("-n",
                        "--num_samples", 
                        type=int, 
                        help="The number of samples to be generated.", 
                        default=100)
    parser.add_argument("--corr_timesteps",
                        nargs="*",
                        type=int, 
                        help="The timesteps at which the corrector is applied.", 
                        default=[10, 20, 30])
    parser.add_argument("-m", "--corrector_mask_mode", type=str, default="thresh",
                        choices=CorrectorMaskingMode.get_values())
    parser.add_argument("-th", "--corrector_mask_threshold", type=float, default=0.7)
    parser.add_argument("--save_dir", 
                        type=str, 
                        help="A directory path where the results are saved.", 
                        default="generation_process_results")
    parser.add_argument("--save_images", 
                        action="store_true",
                        help="Save layout images or not.")
    args = parser.parse_args()

    gen_config_path = os.path.join(args.gen_job_dir, "config.yaml")
    gen_train_cfg = load_config(gen_config_path)

    if args.corr_job_dir is not None:
        corr_config_path = os.path.join(args.corr_job_dir, "config.yaml")
        corr_train_cfg = load_config(corr_config_path)
    else:
        corr_train_cfg = None
    
    test_cfg = OmegaConf.structured(TestConfig)
    test_cfg.cond = cond_type

    # initialize data and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(gen_train_cfg, device=device)
    corrector = None if corr_train_cfg is None else build_model(corr_train_cfg, device=device)

    # load ckpt
    model = load_model(
        model=model,
        ckpt_dir=args.gen_job_dir,
        device=device,
        best_or_final="best"
    )
    model.eval()

    if corrector is not None:
        corrector = load_model(
            model=corrector,
            ckpt_dir=args.corr_job_dir,
            device=device,
            best_or_final="best"
        )
        corrector.eval()

    sampling_cfg = OmegaConf.structured(SAMPLING_CONFIG_DICT[test_cfg.sampling])  # NOTE: you may change sampling algorithm
    OmegaConf.set_struct(sampling_cfg, False)

    sampling_cfg = corr_train_cfg.sampling if corrector is not None else gen_train_cfg.sampling
    num_timesteps = gen_train_cfg.backbone.encoder_layer.diffusion_step
    correction_timesteps = [t for t in args.corr_timesteps if t < num_timesteps]
    # TODO: should be configured
    extra_sampling_cfg = dict(
        num_timesteps=num_timesteps,
        corrector_start=-1,
        corrector_end=-1,  
        corrector_steps=1,
        corrector_t_list=correction_timesteps,
        corrector_mask_mode=args.corrector_mask_mode,
        corrector_mask_threshold=args.corrector_mask_threshold,
        use_gumbel_noise=True,
        gumbel_temperature=1.0
    )
    sampling_cfg.update(extra_sampling_cfg)

    gen_train_cfg.dataset.dir = DATASET_DIR
    dataset = instantiate(gen_train_cfg.dataset)(split="test", transform=None)
    save_kwargs = {
        "colors": dataset.colors, 
        "names": dataset.labels,
        "canvas_size": SIZE, 
        "use_grid": True,
        "draw_label": True
    }

    out_dir = args.save_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.save_images:
        image_out_dir = os.path.join(out_dir, "generation_process_images")

    # generate layout
    assert cond_type == "unconditional"
    pad_id, mask_id = model.tokenizer.name_to_id("pad"), model.tokenizer.name_to_id("mask")

    # register configuration
    outputs = dict(
        gen_job_dir=args.gen_job_dir,
        corr_job_dir=args.corr_job_dir,
        pad_id=pad_id,
        mask_id=mask_id,
        num_attributes=model.tokenizer.N_var_per_element,
        num_category=model.tokenizer.N_category,
        num_bbox_bins=model.tokenizer.N_bbox_per_var
    )
    if corrector is not None:
        outputs["corrector_timesteps"] = correction_timesteps

    # generate layout with the original generative model
    orig_ids_for_pkl, orig_preds_for_pkl, orig_preds = run_generation(
        model,
        sampling_cfg=sampling_cfg,
        corrector=None,
        num_samples=args.num_samples
    )
    outputs["original_ids"] = orig_ids_for_pkl
    outputs["original_decoded_ids"] = orig_preds_for_pkl
    # save steps in an image
    if args.save_images:
        orig_image_out_dir = os.path.join(image_out_dir, "original")
        save_generation_process_as_image(
            orig_preds,
            RESULT_SAVE_TIMESTEPS,
            ncol=10,
            out_dir=orig_image_out_dir,
            fname_prefix="original",
            **save_kwargs
        )

    # generate layout with the corrector
    if corrector is not None:
        corr_ids_for_pkl, corr_preds_for_pkl, corr_preds = \
            run_generation(
                model,
                sampling_cfg=sampling_cfg,
                corrector=corrector,
                num_samples=args.num_samples
            )        
        outputs["corrector_ids"] = corr_ids_for_pkl
        outputs["corrector_decoded_ids"] = corr_preds_for_pkl
        if args.save_images:
            corr_image_out_dir = os.path.join(image_out_dir, "corrector")
            save_generation_process_as_image(
                corr_preds,
                RESULT_SAVE_TIMESTEPS,
                corr_timesteps=correction_timesteps,
                ncol=10,
                out_dir=corr_image_out_dir,
                fname_prefix="corrector",
                **save_kwargs
            )

    # save outputs
    result_file = os.path.join(out_dir, "generation_process_output.pickle")
    with open(result_file, "wb") as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    main()