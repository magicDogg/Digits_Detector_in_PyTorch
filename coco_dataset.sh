mkdir coco_data
cd coco_data

mkdir images
cd images

wget -O coco_train.zip "http://images.cocodataset.org/zips/train2017.zip"
wget -O coco_test.zip "http://images.cocodataset.org/zips/test2017.zip"
wget -O coco_valid.zip "http://images.cocodataset.org/zips/val2017.zip"

unzip coco_train.zip
unzip coco_test.zip
unzip coco_valid.zip

rm coco_train.zip
rm coco_test.zip
rm coco_valid.zip

cd ../

wget -O coco_train_val_annotation.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -O coco_train_val_stuff_annotation.zip http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget -O coco_test_info.zip http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip coco_train_val_annotation.zip
unzip coco_train_val_stuff_annotation.zip
unzip coco_test_info.zip

rm coco_train_val_annotation.zip
rm coco_train_val_stuff_annotation.zip
rm coco_test_info.zip
