import tensorflow as tf
import os
import glob

# --- IMPORTANT: SET THIS PATH CORRECTLY ---
# Option 1: Point to a specific TFRecord file
# tfrecord_file_to_inspect = r'D:\piins\myenv\IVUS\train\your_actual_filename.tfrecord' # REPLACE with an actual filename

# Option 2: Find the first TFRecord file in your train directory
train_dir = r'D:\piins\myenv\IVUS\train' # Path to your training TFRecords
tfrecord_files = glob.glob(os.path.join(train_dir, "*.tfrecord"))
if not tfrecord_files:
    tfrecord_files = glob.glob(os.path.join(train_dir, "*.tfrecords")) # Try other extension

if not tfrecord_files:
    print(f"No TFRecord files found in '{train_dir}'. Please check the path.")
    exit()

tfrecord_file_to_inspect = tfrecord_files[0] # Inspect the first file found
print(f"Inspecting TFRecord file: {tfrecord_file_to_inspect}\n")
# --- END OF PATH SETTING ---


if not os.path.exists(tfrecord_file_to_inspect):
    print(f"Error: TFRecord file not found at '{tfrecord_file_to_inspect}'")
    exit()

# Create a TFRecordDataset
raw_dataset = tf.data.TFRecordDataset(tfrecord_file_to_inspect)

# Take one example from the dataset
for raw_record in raw_dataset.take(1): # Take only the first record
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    print('--- Features found in the TFRecord example: ---')
    feature_keys = list(example.features.feature.keys())
    if not feature_keys:
        print("No features found in this record.")
    else:
        for key in feature_keys:
            print(f"- Key: '{key}'")
            # You can also inspect the type of data if needed
            # feature_type = type(example.features.feature[key].ListFields()[0][1])
            # print(f"  Type hint: {feature_type}") 
    print('-----------------------------------------------')
    break # We only need to inspect one record
