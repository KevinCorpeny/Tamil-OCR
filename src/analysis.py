# Import required libraries
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import matplotlib.pyplot as plt
import os
import multiprocessing
import platform

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Set number of CPU cores to use (leave one core free for system)
NUM_CORES = max(1, multiprocessing.cpu_count() - 1)
print(f"Using {NUM_CORES} CPU cores")
print(f"Operating System: {platform.system()}")

# Custom metrics for model evaluation
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Load and preprocess data
def load_data(file_path):
    """Load data from HDF5 file and split into train/validation/test sets"""
    with h5py.File(file_path, 'r') as hdf:
        # Load training data
        x_train = np.array(hdf['Train Data']['x_train'])
        y_train = np.array(hdf['Train Data']['y_train'])
        
        # Load test data
        x_test = np.array(hdf['Test Data']['x_test'])
        y_test = np.array(hdf['Test Data']['y_test'])
        
        # Split training data into train and validation
        x_val = x_train[-7870:]
        y_val = y_train[-7870:]
        x_train = x_train[:-7870]
        y_train = y_train[:-7870]
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def preprocess_data(x_train, x_val, x_test):
    """Preprocess the data: normalize and reshape"""
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = x_train.reshape(-1, 64, 64, 1)
    x_val = x_val.reshape(-1, 64, 64, 1)
    x_test = x_test.reshape(-1, 64, 64, 1)
    
    return x_train, x_val, x_test

def create_model(input_shape=(64, 64, 1), num_classes=156):
    """Create and return the CNN model"""
    model = models.Sequential([
        # First conv block
        layers.Conv2D(64, (3,3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(256, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes)
    ])
    return model

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Check for GPU availability
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Load and preprocess data
    print("Loading data...")
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'HDF5', 'hdf5_uTHCD_compressed.h5')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(file_path)
    x_train, x_val, x_test = preprocess_data(x_train, x_val, x_test)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Test data shape: {x_test.shape}")

    # Set up data augmentation
    print("\nSetting up data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=5,       # Random rotation up to 5 degrees
        width_shift_range=0.1,  # Random horizontal shift up to 10%
        height_shift_range=0.1, # Random vertical shift up to 10%
        zoom_range=0.1,         # Random zoom up to 10%
        fill_mode='nearest',    # Fill shifted pixels with nearest value
        horizontal_flip=False,  # No horizontal flip as it might change character meaning
        vertical_flip=False,    # No vertical flip as it might change character meaning
        validation_split=0.0    # No validation split as we already have validation data
    )
    datagen.fit(x_train)

    # Create and compile model
    print("\nCreating and compiling model...")
    model = create_model()
    
    # Learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', f1_m, precision_m, recall_m]
    )
    model.summary()

    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    # Calculate steps per epoch and batch size
    batch_size = 128  # Reduced batch size for better stability
    steps_per_epoch = len(x_train) // batch_size
    print(f"\nTraining data size: {len(x_train)}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")

    # Create data generators for training and validation
    train_generator = datagen.flow(
        x_train, 
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    # Train the model
    print("\nStarting training...")
    if platform.system() == 'Windows':
        # On Windows, use single process mode to avoid pickling issues
        print("Windows detected - using single process mode")
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=20,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            shuffle=True
        )
    else:
        # On other systems, try multiprocessing
        try:
            history = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=20,
                validation_data=(x_test, y_test),
                callbacks=callbacks,
                workers=NUM_CORES,
                use_multiprocessing=True,
                max_queue_size=10,
                shuffle=True
            )
        except Exception as e:
            print(f"Error during training with multiprocessing: {e}")
            print("Falling back to single process training...")
            history = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=20,
                validation_data=(x_test, y_test),
                callbacks=callbacks,
                shuffle=True
            )

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    print("\nEvaluating model on test set...")
    if platform.system() == 'Windows':
        # On Windows, use single process mode
        test_loss, test_acc, test_f1, test_precision, test_recall = model.evaluate(
            x_test, y_test,
            batch_size=batch_size
        )
    else:
        # On other systems, try multiprocessing
        try:
            test_loss, test_acc, test_f1, test_precision, test_recall = model.evaluate(
                x_test, y_test,
                batch_size=batch_size,
                workers=NUM_CORES,
                use_multiprocessing=True
            )
        except Exception as e:
            print(f"Error during test evaluation with multiprocessing: {e}")
            print("Falling back to single process evaluation...")
            test_loss, test_acc, test_f1, test_precision, test_recall = model.evaluate(
                x_test, y_test,
                batch_size=batch_size
            )
    
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')

    print("\nEvaluating model on validation set...")
    if platform.system() == 'Windows':
        # On Windows, use single process mode
        val_loss, val_acc, val_f1, val_precision, val_recall = model.evaluate(
            x_val, y_val,
            batch_size=batch_size
        )
    else:
        # On other systems, try multiprocessing
        try:
            val_loss, val_acc, val_f1, val_precision, val_recall = model.evaluate(
                x_val, y_val,
                batch_size=batch_size,
                workers=NUM_CORES,
                use_multiprocessing=True
            )
        except Exception as e:
            print(f"Error during validation evaluation with multiprocessing: {e}")
            print("Falling back to single process evaluation...")
            val_loss, val_acc, val_f1, val_precision, val_recall = model.evaluate(
                x_val, y_val,
                batch_size=batch_size
            )
    
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}')
    print(f'Validation F1 Score: {val_f1:.4f}')
    print(f'Validation Precision: {val_precision:.4f}')
    print(f'Validation Recall: {val_recall:.4f}')

if __name__ == "__main__":
    # Set TensorFlow to use all available CPU cores
    tf.config.threading.set_inter_op_parallelism_threads(NUM_CORES)
    tf.config.threading.set_intra_op_parallelism_threads(NUM_CORES)
    
    main()