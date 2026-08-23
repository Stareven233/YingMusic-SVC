#!/bin/bash

source='path/to/source_vocal.wav'
accompany='path/to/accompany.wav'
target="path/to/target_timbre.wav"
diffusion_step=100
fp16="True"
config='./configs/YingMusic-SVC.yml'
checkpoint='path/to/ckpt.pth'
expname="your_exp_name"
cuda="0"


python my_inference.py \
    --source "${source}" \
    --target "${target}" \
    --diffusion-steps "${diffusion_step}" \
    --checkpoint "${checkpoint}" \
    --expname "${expname}" \
    --cuda "${cuda}" \
    --fp16 "${fp16}" \
    --accompany "${accompany}" \
    --config "${config}"
