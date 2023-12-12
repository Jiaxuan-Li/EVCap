CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 ./train_evcap.py \
    --out_dir results/train_evcap