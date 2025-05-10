export CUDA_VISIBLE_DEVICES=0

python ./zeroshot/inference_endoscope_testset.py \
--base-dir /data_new/luxiaoxi/code_proj/MedicalMast3R \
--input-dir /data_new/luxiaoxi/dataset/medical_depth/SCARED_keyframe_nearest_256 \
--output-dir /data_new/luxiaoxi/dataset/medical_depth_output/endoscope_mast3r_unfreeze_decoder_encoder_0510 \
--data-name scared \
--model-name checkpoints/together_unfreeze_head_decoder_0510/checkpoint-best.pth

python ./zeroshot/inference_endoscope_testset.py \
--base-dir /data_new/luxiaoxi/code_proj/MedicalMast3R \
--input-dir /data_new/luxiaoxi/dataset/medical_depth/EndoAbs_preprocessed_nearest \
--output-dir /data_new/luxiaoxi/dataset/medical_depth_output/endoscope_mast3r_unfreeze_decoder_encoder_0510 \
--data-name abs \
--model-name checkpoints/together_unfreeze_head_decoder_0510/checkpoint-best.pth

python ./zeroshot/inference_endoscope_testset.py \
--base-dir /data_new/luxiaoxi/code_proj/MedicalMast3R \
--input-dir /data_new/luxiaoxi/dataset/medical_depth/SERV-CT_preprocessed \
--output-dir /data_new/luxiaoxi/dataset/medical_depth_output/endoscope_mast3r_finetune_nofreeze \
--data-name servct \
--model-name checkpoints/together_unfreeze_head_decoder_0510/checkpoint-best.pth
