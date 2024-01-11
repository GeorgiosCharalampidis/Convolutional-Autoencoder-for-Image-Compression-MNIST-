import numpy as np
import struct
import argparse
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Dropout, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def read_mnist_file(file_path):
    with open(file_path, 'rb') as file:
        magic, num_images, rows, cols = struct.unpack('>IIII', file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def preprocess_images(images):
    images = images.astype('float32') / 255.
    images = np.reshape(images, (len(images), 28, 28, 1))
    return images

def train_autoencoder(input_images, conv_layers=3, filter_size=(3, 3), filters_per_layer=64, epochs=20, batch_size=128, latent_dim=20):
    x_train, x_val = train_test_split(input_images, test_size=0.2, random_state=42)

    input_img = Input(shape=(28, 28, 1))
    x = input_img

    # Encoder
    for _ in range(conv_layers):
        x = Conv2D(filters_per_layer, filter_size, padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Dropout(0.4)(x)

    # Flatten and encode
    x = Flatten()(x)
    x = Dense(latent_dim, activation='relu')(x)
    encoded = x

    # Decoder
    x = Dense(7 * 7 * filters_per_layer, activation='relu')(encoded)
    x = Reshape((7, 7, filters_per_layer))(x)

    # Upsampling to restore the dimensions
    for _ in range(conv_layers - 1):
        x = Conv2D(filters_per_layer, filter_size, padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)
        x = Dropout(0.4)(x)

    # Final Conv2D layer to match the input shape
    x = Conv2D(1, filter_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, x)
    encoder = Model(input_img, encoded)

    opt = Adam(learning_rate=0.0005)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)

    autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_val, x_val), callbacks=[early_stop, lr_scheduler])

    return autoencoder, encoder

def normalize_vectors(vectors):
    vectors = (vectors - vectors.min()) / (vectors.max() - vectors.min())
    return (vectors * 255).astype(np.uint8)

def save_encoded_vectors(encoded_vectors, file_path, latent_dim=10):
    # New magic number for encoded vectors (choose a unique number)
    magic_num = 3051  # Different from original MNIST magic numbers to avoid confusion
    number_of_vectors = encoded_vectors.shape[0]

    # Instead of rows and columns, we use latent_dim twice to maintain the header format
    # This indicates that each 'image' is now a vector of size latent_dim
    vector_dim1 = vector_dim2 = latent_dim

    with open(file_path, 'wb') as file:
        file.write(magic_num.to_bytes(4, byteorder='big'))
        file.write(number_of_vectors.to_bytes(4, byteorder='big'))
        file.write(vector_dim1.to_bytes(4, byteorder='big'))
        file.write(vector_dim2.to_bytes(4, byteorder='big'))
        # Write the encoded vectors
        encoded_vectors.astype(np.uint8).tofile(file)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input_data', required=True, help="Path to the input data file")
    parser.add_argument('-q', '--query_data', required=True, help="Path to the query data file")
    parser.add_argument('-od', '--output_data', required=True, help="Path to the output data file")
    parser.add_argument('-oq', '--output_query', required=True, help="Path to the output query file")
    args = parser.parse_args()

    input_images = preprocess_images(read_mnist_file(args.input_data))
    query_images = preprocess_images(read_mnist_file(args.query_data))

    autoencoder, encoder = train_autoencoder(input_images)

    # Save the autoencoder and encoder models
    autoencoder.save('autoencoder_model.keras')
    encoder.save('encoder_model.keras')

    # Getting encoded vectors
    encoded_input = encoder.predict(input_images)
    encoded_query = encoder.predict(query_images)

    normalized_input = normalize_vectors(encoded_input)
    normalized_query = normalize_vectors(encoded_query)

    save_encoded_vectors(normalized_input, args.output_data, latent_dim=20)
    save_encoded_vectors(normalized_query, args.output_query, latent_dim=20)

if __name__ == '__main__':
    main()
