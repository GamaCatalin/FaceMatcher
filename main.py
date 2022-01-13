# Import standard dependencies
import cv2
import os
import random
import numpy as np
from keras.metrics import Recall, Precision
from matplotlib import pyplot as plt
# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import uuid

# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
MODEL_PATH = "siamesemodelv2.h5"


# # enable GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


# =========================== SETUP =====================================

def setup_negative_images():
    for directory in os.listdir('lfw'):
        for file in os.listdir(os.path.join('lfw', directory)):
            EX_PATH = os.path.join('lfw', directory, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)


def webcam_capture():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        # Cut down frame to 250x250px
        frame = frame[120:120 + 250, 200:200 + 250, :]

        # Collect anchors
        if cv2.waitKey(1) & 0XFF == ord('a'):
            # Create the unique file path
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)

        # Collect positives
        if cv2.waitKey(1) & 0XFF == ord('p'):
            # Create the unique file path
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out positive image
            cv2.imwrite(imgname, frame)

        # Show image back to screen
        cv2.imshow('Image Collection', frame)

        # Breaking gracefully
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()


# =========================== DATASET =====================================

def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


def create_dataset():
    anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(300)
    positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(300)
    negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(300)

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    # Build dataloader pipeline
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)

    return data


def get_training_data(data):
    # Training partition
    train_data = data.take(round(len(data) * .7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    return train_data


def get_test_data(data):
    # Testing partition
    test_data = data.skip(round(len(data) * .7))
    test_data = test_data.take(round(len(data) * .3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    return test_data


# =========================== MODEL =====================================


def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


class L1Dist(Layer):
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)



def make_siamese_model(embedding):
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

# =========================== TRAINING =====================================

def do_training(siamese_model):
    # loss and optimizer
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001

    # checkpoint callback
    checkpoint_dir = './training_checkpoints2'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

    # init datasets
    dataset = create_dataset()
    train_data = get_training_data(dataset)

    # train on one batch
    @tf.function
    def train_step(batch):
        # Record all of our operations
        with tf.GradientTape() as tape:
            # Get anchor and positive/negative image
            X = batch[:2]
            # Get label
            y = batch[2]

            # Forward pass
            yhat = siamese_model(X, training=True)
            # Calculate loss
            loss = binary_cross_loss(y, yhat)
        print(loss)

        # Calculate gradients
        grad = tape.gradient(loss, siamese_model.trainable_variables)

        # Calculate updated weights and apply to siamese model
        opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

        # Return loss
        return loss

    # training function
    def train(data, epochs):
        # Loop through epochs
        for epoch in range(1, epochs + 1):
            print(f'\n Epoch {epoch}/{epochs}')
            progbar = tf.keras.utils.Progbar(len(data))

            # Creating a metric object
            r = Recall()
            p = Precision()
            loss = []
            # Loop through each batch
            for idx, batch in enumerate(data):
                # Run train step here
                loss = train_step(batch)
                yhat = siamese_model.predict(batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                progbar.update(idx + 1)
            print(loss.numpy(), r.result().numpy(), p.result().numpy())

            # Save checkpoints
            if epoch % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                save_model(siamese_model, MODEL_PATH)
            # checkpoint.save(file_prefix=checkpoint_prefix)
            save_model(siamese_model, MODEL_PATH)

    EPOCHS = 50
    train(train_data, EPOCHS)  # train the model


# =========================== EVALUATE =====================================


def get_recall(y_true, y_hat):
    # Creating a metric object
    m = Recall()

    # Calculating the recall value
    m.update_state(y_true, y_hat)

    # Return Recall Result
    return m.result().numpy()


def get_precision(y_true, y_hat):
    # Creating a metric object
    m = Precision()

    # Calculating the recall value
    m.update_state(y_true, y_hat)

    # Return Recall Result
    return m.result().numpy()


def check_prediction(model, test_data):
    r = Recall()
    p = Precision()
    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = model.predict([test_input, test_val])
        r.update_state(y_true, yhat)
        p.update_state(y_true, yhat)
    print(f"recall: {r.result().numpy()}, precision {p.result().numpy()}")


def show_test_img(img_input, img_val):
    # Set plot size
    plt.figure(figsize=(10, 8))

    # Set first subplot
    plt.subplot(1, 2, 1)
    plt.imshow(img_input)

    # Set second subplot
    plt.subplot(1, 2, 2)
    plt.imshow(img_val)

    # Renders cleanly
    plt.show()


# =========================== SAVE MODEL =====================================


def save_model(model, path):
    model.save(path)



def load_model(model_path):
    return tf.keras.models.load_model(model_path,
                                      custom_objects={'L1Dist': L1Dist,
                                                      'BinaryCrossentropy': tf.losses.BinaryCrossentropy})


# =========================== LIVE TESTING =====================================


def setup_model():
    embedding = make_embedding()
    return make_siamese_model(embedding)


def model_from_checkpoint(checkpoint_path):
    opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001
    siamese_model = setup_model()
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
    checkpoint.restore(checkpoint_path)
    return siamese_model


def train_run():
    siamese_model = load_model(MODEL_PATH)
    siamese_model.compile()
    do_training(siamese_model)
    save_model(siamese_model, MODEL_PATH)


def test_run():
    siamese_model = load_model(MODEL_PATH)
    siamese_model.compile()
    siamese_model.summary()
    data = create_dataset()
    test_data = get_test_data(data)
    test_input, test_val, y_true = test_data.as_numpy_iterator().next()
    check_prediction(siamese_model, test_data)

    show_test_img(test_input[0], test_val[0])


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    # for image in os.listdir(os.path.join('img_verification, 'validation_image')):



    input_img = preprocess(os.path.join('img_verification', 'input_image', 'input_image.jpg'))
    validation_img = preprocess(os.path.join('img_verification', 'validation_image', 'validation_image.jpg'))

    # Make Predictions
    result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
    results.append(result)





    # Detection Threshold: Metric above which a prediciton is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    # verification = detection / len(os.listdir(os.path.join('application_data', 'validation_images')))
    verification = detection / len(results)
    verified = verification > verification_threshold

    return results, verified

def rt_run():
    cap = cv2.VideoCapture(0)
    siamese_model = load_model(MODEL_PATH)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]

        cv2.imshow('Verification', frame)


        if cv2.waitKey(10) & 0xFF == ord('v'):
            # Save input image to application_data/input_image folder
            cv2.imwrite(os.path.join('img_verification', 'input_image', 'input_image.jpg'), frame)
            # Run verification
            results, verified = verify(siamese_model, 0.7, 0.7)
            print(verified)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def run():
    # train_run()
    # test_run()
    rt_run()


if __name__ == "__main__":
    run()
