python deploy/ONNX/export_onnx.py \
    --weights ~/autodl-tmp/weights/runs/v6n_coco_multiGPU/weights/best_stop_aug_ckpt.pt \
    --topk-all 100 \
    --img-size 640 640 \
    --dynamic-batch \
    --ort