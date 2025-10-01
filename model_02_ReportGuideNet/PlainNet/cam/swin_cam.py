# cam/swin_cam.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def swin_gradcam_visualize(model, dataloader, device, outdir):
    """
    Swin Transformer + EigenCAM 시각화 (stage2/3/4 target 비교 저장)
    """
    os.makedirs(outdir, exist_ok=True)

    # 여러 후보 target layers 준비
    candidate_layers = {
        "stage2_mlp": [model.stage2.layers[0][-1].mlp_block.fn.fn.net[0]],
        "stage3_attn": [model.stage3.layers[0][-1].attention_block.fn.fn.to_out],
        "stage4_attn": [model.stage4.layers[0][-1].attention_block.fn.fn.to_out],
    }

    model.eval()
    for name, layers in candidate_layers.items():
        print(f"[INFO] Running EigenCAM for {name}")
        cam = EigenCAM(model=model, target_layers=layers)

        subdir = os.path.join(outdir, name)
        os.makedirs(subdir, exist_ok=True)

        for batch in dataloader:
            imgs = batch["image"].to(device)   # (B,1,H,W)
            labels = batch["label"].to(device)
            sids = batch["sid"]
            slice_idxs = batch["slice_idx"]

            # 입력 정규화
            imgs_norm = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-6)

            # 모델 예측
            outputs = model(imgs_norm)
            preds = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1)

            for i in range(len(imgs)):
                grayscale_cam = cam(
                    input_tensor=imgs_norm[i:i+1],
                    targets=None
                )[0]  # (H,W)

                # 원본 slice
                img_np = imgs_norm[i][0].cpu().numpy()
                rgb_img = np.stack([img_np]*3, axis=-1)

                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                sid = sids[i]
                prob = preds[i]
                true_label = int(labels[i].cpu().item())
                pred_label = int(prob >= 0.5)

                filename = f"{sid}_slice{slice_idxs[i]:03d}_true{true_label}_pred{pred_label}_prob{prob:.3f}.png"
                out_path = os.path.join(subdir, filename)

                plt.imsave(out_path, visualization)
