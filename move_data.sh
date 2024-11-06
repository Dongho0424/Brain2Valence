#!/bin/bash

# Create directories if they don't exist
mkdir -p /home/dongho/brain2valence/data/vis
mkdir -p /home/dongho/brain2valence/data/emo
mkdir -p /home/dongho/brain2valence/data/emo_vis

for s in 1 2 5 7
do
   cp /home/juhyeon/Brain2Valence/vis_subj${s}_all_beta.npy /home/dongho/brain2valence/data/vis/
   cp /home/juhyeon/Brain2Valence/vis_subj${s}_train_beta_mean.npy /home/dongho/brain2valence/data/vis/
   cp /home/juhyeon/Brain2Valence/vis_subj${s}_train_beta_std.npy /home/dongho/brain2valence/data/vis/
   cp /home/juhyeon/Brain2Valence/emo_subj${s}_all_beta.npy /home/dongho/brain2valence/data/emo/
   cp /home/juhyeon/Brain2Valence/emo_subj${s}_train_beta_mean.npy /home/dongho/brain2valence/data/emo/
   cp /home/juhyeon/Brain2Valence/emo_subj${s}_train_beta_std.npy /home/dongho/brain2valence/data/emo/
   cp /home/juhyeon/Brain2Valence/emo_vis_subj${s}_all_beta.npy /home/dongho/brain2valence/data/emo_vis/
   cp /home/juhyeon/Brain2Valence/emo_vis_subj${s}_train_beta_mean.npy /home/dongho/brain2valence/data/emo_vis/
   cp /home/juhyeon/Brain2Valence/emo_vis_subj${s}_train_beta_std.npy /home/dongho/brain2valence/data/emo_vis/
done