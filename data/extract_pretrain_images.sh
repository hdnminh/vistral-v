#!/bin/bash

# export DATA_DIR=/mnt/disks/dev/data/images

# LLAVA
mkdir ${DATA_DIR}/llava/llava_pretrain
mkdir ${DATA_DIR}/llava/llava_pretrain/images
unzip ${DATA_DIR}/llava/images.zip -d ${DATA_DIR}/llava/llava_pretrain/images
rm ${DATA_DIR}/llava/images.zip

# SAM
cat ${DATA_DIR}/sam/images_part_* | tar -xvzf - -C ${DATA_DIR}/sam
rm ${DATA_DIR}/sam/images_part_*

# share_textvqa
unzip -j ${DATA_DIR}/share_textvqa/images.zip -d ${DATA_DIR}/share_textvqa/images
rm ${DATA_DIR}/share_textvqa/images.zip

# web-celebrity
unzip -j ${DATA_DIR}/web-celebrity/images.zip -d ${DATA_DIR}/web-celebrity/images
rm ${DATA_DIR}/web-celebrity/images.zip 

# web-landmark
unzip -j ${DATA_DIR}/web-landmark/images.zip -d ${DATA_DIR}/web-landmark/images
rm ${DATA_DIR}/web-landmark/images.zip

# wikiart
unzip -j ${DATA_DIR}/wikiart/images.zip -d ${DATA_DIR}/wikiart/images
rm ${DATA_DIR}/wikiart/images.zip

# WIT
cat ${DATA_DIR}/wit/images_part_* | tar -xvzf - -C ${DATA_DIR}/wit
rm ${DATA_DIR}/wit/images_part_*