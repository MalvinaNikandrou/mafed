DATA_DIR=storage/data
IMAGES_DIR=$DATA_DIR/images
VQA_DIR=$DATA_DIR/VQA
CONTVQA_DIR=$VQA_DIR/contvqa

# COCO IMAGES
mkdir -p $IMAGES_DIR
wget http://images.cocodataset.org/zips/train2014.zip -P $DATA_DIR
unzip $DATA_DIR/train2014.zip -d $IMAGES_DIR
wget http://images.cocodataset.org/zips/val2014.zip -P $DATA_DIR
unzip $DATA_DIR/val2014.zip -d $IMAGES_DIR

# VQA DATA
mkdir -p $VQA_DIR
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P $DATA_DIR
unzip $DATA_DIR/v2_Questions_Train_mscoco.zip -d $VQA_DIR
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P $DATA_DIR
unzip $DATA_DIR/v2_Questions_Val_mscoco.zip -d $VQA_DIR

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P $DATA_DIR
unzip $DATA_DIR/v2_Annotations_Train_mscoco.zip -d $VQA_DIR
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P $DATA_DIR
unzip $DATA_DIR/v2_Annotations_Val_mscoco.zip -d $VQA_DIR

# CONTVQA DATA
git clone --separate-git-dir=$(mktemp -u) --depth=1 https://github.com/MalvinaNikandrou/contvqa.git $CONTVQA_DIR && rm $CONTVQA_DIR/.git
python mafed/data/preprocess.py --data_dir $DATA_DIR