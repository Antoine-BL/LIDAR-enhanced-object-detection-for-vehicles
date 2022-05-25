#Init gcloud
$project_id = 'splendid-binder-350320'
gcloud config set project $project_id

# Get Directories Ready
New-Item -ItemType Directory -Force -Path '..\data\training\tar'
New-Item -ItemType Directory -Force -Path '..\data\validation\tar'

# Download training data
for (($i = 0); $i -lt 32; $i++) {
    $file_name = 'training_' + $i.ToString().PadLeft(4, '0') + '.tar'
    gsutil cp gs://waymo_open_dataset_v_1_1_0/training/$file_name  ../data/training/tar/$file_name
}

# Download validation data
for (($i = 0); $i -lt 8; $i++) {
    $file_name = 'validation_' + $i.ToString().PadLeft(4, '0') + '.tar'
    gsutil cp gs://waymo_open_dataset_v_1_1_0/validation/$file_name ../data/validation/tar/$file_name
}