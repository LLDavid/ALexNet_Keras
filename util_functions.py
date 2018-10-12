import tensorflow as tf
## Read all images in folder into tensor
import numpy as np

def Indices2OneHot(class_indices):
    max_i=np.max(class_indices)+1
    class_labels=np.zeros([np.size(class_indices,0),max_i])
    for i in range(np.size(class_indices,0)):
        class_labels[i][class_indices[i]]=1
    return class_labels

def read_allimg_folder(file_dir, type):
    """
    :param file_dir: "path/to/folder"
    :param type:  "JPG"
    :return: tensor of all images
    """
    file_path=file_dir+"/."+type
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(file_path))
    # Read an entire image file which is required since they're JPEGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    _, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image = tf.image.decode_jpeg(image_file)

    return image

## print layer info
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

## Network model
def alex_net(images):
    """Build the AlexNet model.
    Args:
    images: Images Tensor
    Returns:
    pool5: the last Tensor in the convolutional component of AlexNet.
    parameters: a list of Tensors corresponding to the weights and biases of the
        AlexNet model.
    """
    parameters = []
    ## conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                             stddev=1e-1), name='weights') # Mean = 0.0 by default
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME') # Stides: [1, 4, 4, 1] -> [batch, h, w, channel]
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases) # biases matching the last dimension of values (1st arg): conv
        conv1 = tf.nn.relu(bias, name=scope)
    print_activations(conv1)
    parameters += [kernel, biases]

    ## lrn1
    lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')

    ## pool1
    pool1 = tf.nn.max_pool(lrn1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')
    print_activations(pool1)

    ## conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv2)

    ## pool2
    pool2 = tf.nn.max_pool(conv2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding="SAME",
                         name='pool2')
    print_activations(pool2)

    ## conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)

    ## conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)

    ## conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)

    ## pool5
    pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME',
                         name='pool5')
    print_activations(pool5)

    # ## Flatten
    pool5=tf.contrib.layers.flatten(pool5)
    #pool5 = tf.manip.reshape(pool5, [-1])

    ## FC1
    fc1=tf.layers.dense(pool5, 4096, activation=tf.nn.relu,  trainable=True)
    print_activations(fc1)

    ## FC2
    fc2=tf.layers.dense(fc1, 4096, activation=tf.nn.relu,  trainable=True)
    print_activations(fc2)

    ## Ouput
    out1=tf.layers.dense(fc2, 10, activation=None,  trainable=True)

    return out1, parameters

