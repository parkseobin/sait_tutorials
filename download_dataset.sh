cd pix2pix
mkdir datasets
FILE=facades
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
cd ..

cd cyclegan
mkdir datasets
FILE=horse2zebra
if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "ae_photos" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
cd ..

cd meta-learning/data
wget https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true
wget https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true
mv images_background.zip?raw=true images_background.zip
mv images_evaluation.zip?raw=true images_evaluation.zip
unzip -q images_background.zip
unzip -q images_evaluation.zip
mv images_background/* omniglot_resized
mv images_evaluation/* omniglot_resized
rmdir images_evaluation images_background
cd omniglot_resized
python resize_images.py
cd ..
rm -rf *.zip
cd ../..


