import h5py
import numpy as np
from pathlib import Path
import tensorflow as tf
import torch
from tqdm import tqdm
import os
import sys
import glob
import random
import pdb
np.random.seed(0)
generate_tf_record = False

tfrecord_path = "/path/to/save/your/tfrecord/"
path_to_wavlm_feat = "/path/to/your/wavlm/feat"

if not os.path.exists(tfrecord_path):
    generate_tf_record = True
os.makedirs(tfrecord_path, exist_ok=True)
train_filename = tfrecord_path + 'train'
valid_filename= tfrecord_path + 'valid'
test_filename= tfrecord_path + 'test'
train_path = Path(os.path.join(path_to_wavlm_feat, "train-clean-100"))
valid_path = Path(os.path.join(path_to_wavlm_feat, "dev-clean"))
test_path = Path(os.path.join(path_to_wavlm_feat, "test-clean"))

train_size = 27269
valid_size = 1940
test_size = 1850

def get_filenames(path):
    all_files = []
    all_files.extend(list(path.rglob("**/*.pt")))
    return all_files

def length_filter(paths):
    filtered_paths = []
    print("filter short files")
    for each in tqdm(paths):
        data = torch.load(each).numpy().astype(np.float32)
        if data.shape[0] < 200:
            continue
        filtered_paths.append(each)
    return filtered_paths


def generate_mask(x, mask_type):
    if mask_type == b'expand':
        m = np.zeros_like(x)
        N = np.random.randint(x.shape[0]//8, x.shape[0])
        ind = np.random.choice(x.shape[0], N, replace=False)
        m[ind] = 1.
    elif mask_type == b'few_expand':
        m = np.zeros_like(x)
        N = np.random.randint(x.shape[0]//8)
        ind = np.random.choice(x.shape[0], N, replace=False)
        m[ind] = 1.
    elif mask_type == b'arb_expand':
        m = np.zeros_like(x)
        N = np.random.randint(x.shape[0])
        ind = np.random.choice(x.shape[0], N, replace=False)
        m[ind] = 1.
    elif mask_type == b'det_expand':
        m = np.zeros_like(x)
        ind = np.random.choice(x.shape[0], 100, replace=False)
        m[ind] = 1.
    elif mask_type == b'complete':
        m = np.zeros_like(x)
        while np.sum(m[:,0]) < x.shape[0] // 8:
            p = np.random.uniform(-0.5, 0.5, size=4)
            xa = np.concatenate([x, np.ones([x.shape[0],1])], axis=1)
            m = (np.dot(xa, p) > 0).astype(np.float32)
            m = np.repeat(np.expand_dims(m, axis=1), 3, axis=1)
    else:
        raise ValueError()

    return m


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()
def convert(image_paths, out_path, max_files=1000):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.
    
    print("Converting: " + out_path)
    
    # Number of images. Used when printing the progress.
    num_images = len(image_paths)
    splits = (num_images//max_files) + 1 
    if num_images%max_files == 0:
        splits-=1
    print(f"\nUsing {splits} shard(s) for {num_images} files, with up to {max_files} samples per shard")
    file_count = 0
    for i in tqdm(range(splits)):
        # Open a TFRecordWriter for the output-file.
        with tf.python_io.TFRecordWriter("{}_{}_{}.tfrecords".format(out_path, i+1, splits)) as writer:
            
            # Iterate over all the image-paths and class-labels.
            current_shard_count = 0
            while current_shard_count < max_files: 
                index = i*max_files+current_shard_count
                if index == len(image_paths):
                    break
                current_image = image_paths[index]

                # Load the image-file using matplotlib's imread function.
                img = torch.load(current_image).numpy().astype(np.float32)
                
                # Convert the image to raw bytes.
                img_bytes = img.tostring()

                # Create a dict with the data we want to save in the
                # TFRecords file. You can add more relevant data here.
                data = \
                    {
                        'image': wrap_bytes(img_bytes),
                        'length': wrap_int64(img.shape[0]),
                        "filename": wrap_bytes(bytes(os.path.splitext(current_image.name)[0], 'utf-8'))
                    }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()
                
                # Write the serialized data to the TFRecords file.
                writer.write(serialized)
                current_shard_count+=1
                file_count += 1
    print(f"\nWrote {file_count} elements to TFRecord")


if generate_tf_record:
    train_image_paths = length_filter(get_filenames(train_path))
    valid_image_paths = length_filter(get_filenames(valid_path))
    test_image_paths = length_filter(get_filenames(test_path))
    print(f"Number of training data after length filering: {len(train_image_paths)}")
    print(f"Number of valid data after length filering: {len(valid_image_paths)}")
    print(f"Number of testing data after length filering: {len(test_image_paths)}")
    random.Random(4).shuffle(train_image_paths)

    train_size = len(train_image_paths)
    valid_size = len(valid_image_paths)
    test_size = len(test_image_paths)
    convert(image_paths=train_image_paths,
            out_path=train_filename)

    convert(image_paths=valid_image_paths,
            out_path=valid_filename)

    convert(image_paths=test_image_paths,
            out_path=test_filename)
    

def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'length': tf.FixedLenFeature([], tf.int64),
            'filename': tf.FixedLenFeature([], tf.string),
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.float32)
    

    # Get the label associated with the image.
    length = parsed_example['length']

    image = tf.reshape(image, [length, 1024])
    filename = parsed_example['filename']

    # The image and label are now correct TensorFlow types.
    return image, filename

def process(x, filename, set_size, mask_type):
    x = x/10
    ind = np.random.choice(x.shape[0], set_size, replace=False)
    x = x[ind] 
    m = generate_mask(x, mask_type)
    #N = np.random.randint(set_size)
    #S = np.random.randint(x.shape[0] - set_size + 1)
    #x = x[S:S+set_size]
    #m = np.zeros_like(x)
    #S = np.random.randint(set_size - N + 1)
    #m[S:S+N] = 1.0
    return x, m, filename


def get_dst(split, set_size, mask_type):
    if split == 'train':
        files = glob.glob(train_filename+"*.tfrecords", recursive=False)
        dst = tf.data.TFRecordDataset(files)
        size = train_size
        dst = dst.map(parse)
        dst = dst.shuffle(256)
        dst = dst.map(lambda x, y: tuple(tf.py_func(process, [x, y, set_size, mask_type], [tf.float32, tf.float32, tf.string])), num_parallel_calls=8)
    elif split == 'valid':
        files = glob.glob(valid_filename+"*.tfrecords", recursive=False)
        dst = tf.data.TFRecordDataset(files)
        size = valid_size
        dst = dst.map(parse)
        dst = dst.map(lambda x, y: tuple(tf.py_func(process, [x, y, set_size, mask_type], [tf.float32, tf.float32, tf.string])), num_parallel_calls=8)
    else:
        files = glob.glob(test_filename+"*.tfrecords", recursive=False)
        dst = tf.data.TFRecordDataset(files)
        size = test_size
        dst = dst.map(parse)
        dst = dst.map(lambda x, y: tuple(tf.py_func(process, [x, y, set_size, mask_type], [tf.float32, tf.float32, tf.string])), num_parallel_calls=8)
    return dst, size

class Dataset(object):
    def __init__(self, split, batch_size, set_size, mask_type):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            dst, size = get_dst(split, set_size, mask_type)
            self.size = size
            self.num_batches = self.size // batch_size
            dst = dst.batch(batch_size, drop_remainder=False)
            dst = dst.prefetch(1)

            dst_it = dst.make_initializable_iterator()
            x, b, filename  = dst_it.get_next()
            self.x = x
            self.b = b
            self.filename = filename
            #self.x = tf.reshape(x, [batch_size, set_size, 1024])
            #self.b = tf.reshape(b, [batch_size, set_size, 1024])
            self.dimension = 1024
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        x, b, filename = self.sess.run([self.x, self.b, self.filename])
        m = np.ones_like(b)
        return {'x':x, 'b':b, 'm':m, "f":filename}