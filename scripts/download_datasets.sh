#!/usr/bin/env bash
set -e

mkdir -p data/mot17
mkdir -p data/duke
cd data

#####################################
# MOT17 official train set
#####################################

if [ ! -d "mot17/MOT17-02/img1" ]; then
  echo "Downloading MOT17 train..."
  wget -c https://motchallenge.net/data/MOT17.zip -O MOT17.zip
  
  echo "Extracting MOT17..."
  unzip -q MOT17.zip
  rm MOT17.zip
  
  mv MOT17/train/MOT17-02 mot17/
  rm -rf MOT17
fi
echo "Datasets ready."