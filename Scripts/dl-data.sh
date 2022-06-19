#Init gcloud
project_id='splendid-binder-350320'
gcloud config set project $project_id

# Download training data
for i in $(seq -f "%04g" 16 31)
do
    file_name=training_$i.tar
    gsutil cp gs://waymo_open_dataset_v_1_1_0/training/$file_name  ../Data/training/tar/$file_name
done

# Download validation data
for i in $(seq -f "%04g" 0 7)
do
    file_name=validation_$i.tar
    gsutil cp gs://waymo_open_dataset_v_1_1_0/validation/$file_name ../Data/validation/tar/$file_name
done
