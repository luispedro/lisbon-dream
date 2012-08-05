#!/usr/bin/env bash

mkdir -p data
cd data
DREAM_USERNAME=ncidream
DREAM_PASSWORD=SpqjKd4C
wget ftp://$DREAM_USERNAME:$DREAM_PASSWORD@ftp-private.ebi.ac.uk/upload/data/DREAM7_DrugSensitivity1.zip
wget ftp://$DREAM_USERNAME:$DREAM_PASSWORD@ftp-private.ebi.ac.uk/upload/data/DREAM7_DrugSensitivity2.zip

unzip DREAM7_DrugSensitivity1.zip
mkdir DrugSensitivity2
cd DrugSensitivity2
mv ../DREAM7_DrugSensitivity2.zip .
unzip DREAM7_DrugSensitivity2.zip
mv DREAM7_DrugSensitivity2.zip ..
cd ..
rm -rf __MACOSX/
