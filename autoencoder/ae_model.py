import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Параметры модели
# input_dim = 40  # 40 числовых признаков
encoding_dim = 16  # Размер скрытого представления

class Encoder:
    def __init__(self, encoding_dim: int, input_dim: int):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

    def get_model(self):
        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.input_dim,)),
            tf.keras.layers.Dense(128, activation='relu',  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.encoding_dim, activation='relu')
        ])
        return encoder

class Decoder:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def get_model(self):
        decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),

            tf.keras.layers.Dense(24, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.input_dim, activation='linear')
        ])
        return decoder

class Autoencoder(Model):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoding_dim, input_dim).get_model()
        self.decoder = Decoder(input_dim).get_model()

    def call(self, inputs, training=None, mask=None):
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def decode(self, encoded, training=False):
        return self.decoder(encoded, training=training)


# Функция для создания и обучения автоэнкодера
def create_and_train_autoencoder(X_train, X_val, input_dim, encoding_dim, epochs=100):
    """
    Создание и обучение автоэнкодера
    """
    # Создаем автоэнкодер
    autoencoder = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)

    # Компилируем модель
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint('best_autoencoder.keras', save_best_only=True, monitor='val_loss')
    ]

    # Обучение
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1
    )

    return autoencoder, history


# Функция для визуализации результатов обучения
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Функция для визуализации оригинальных и восстановленных данных
def plot_reconstruction_results(original, reconstructed, num_samples=5):
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    for i in range(num_samples):
        # Оригинальные данные
        axes[0, i].plot(original[i], 'b-', linewidth=2, label='Original')
        axes[0, i].set_title(f'Sample {i + 1}\nOriginal')
        axes[0, i].grid(True, alpha=0.3)

        # Восстановленные данные
        axes[1, i].plot(reconstructed[i], 'r-', linewidth=2, label='Reconstructed')
        axes[1, i].set_title(f'Sample {i + 1}\nReconstructed')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# # Пример использования с синтетическими данными
# if __name__ == "__main__":
#     # Генерация синтетических данных для примера
#     num_samples = 1000
#     X = np.random.randn(num_samples, input_dim).astype(np.float32)
#
#     # Добавляем некоторую структуру для лучшей демонстрации
#     X = X + 0.5 * np.sin(np.arange(input_dim) * 0.5)
#
#     # Разделение на train/validation
#     X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
#
#     # Нормализация данных (важно для автоэнкодеров)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#
#     print(f"Data shapes - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}")
#
#     # Создание и обучение автоэнкодера
#     autoencoder, history = create_and_train_autoencoder(
#         X_train_scaled, X_val_scaled, input_dim, encoding_dim, epochs=50
#     )
#
#     # Визуализация процесса обучения
#     plot_training_history(history)
#
#     # Тестирование обученной модели
#     test_sample = X_val_scaled[:5]
#     reconstructed = autoencoder.predict(test_sample)
#
#     print("\n=== Model Evaluation ===")
#     print("Original data shape:", test_sample.shape)
#     print("Reconstructed data shape:", reconstructed.shape)
#
#     # Вычисление ошибки реконструкции
#     mse = np.mean((test_sample - reconstructed) ** 2)
#     mae = np.mean(np.abs(test_sample - reconstructed))
#     print(f"Reconstruction MSE: {mse:.6f}")
#     print(f"Reconstruction MAE: {mae:.6f}")
#
#     # Получение скрытых представлений
#     encoded_data = autoencoder.encode(test_sample)
#     print(f"Encoded representation shape: {encoded_data.shape}")
#
#     # Визуализация оригинальных и восстановленных данных
#     plot_reconstruction_results(test_sample, reconstructed)
#
#     # Дополнительная визуализация: сравнение первых нескольких признаков
#     plt.figure(figsize=(12, 8))
#     for i in range(min(8, input_dim)):  # Показываем первые 8 признаков
#         plt.subplot(2, 4, i + 1)
#         plt.scatter(test_sample[:, i], reconstructed[:, i], alpha=0.6)
#         plt.plot([test_sample[:, i].min(), test_sample[:, i].max()],
#                  [test_sample[:, i].min(), test_sample[:, i].max()], 'r--', alpha=0.8)
#         plt.xlabel('Original')
#         plt.ylabel('Reconstructed')
#         plt.title(f'Feature {i + 1}')
#         plt.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.show()
#
#     # Сохранение модели
#     autoencoder.save('autoencoder_model.h5')
#     print("\nModel saved as 'autoencoder_model.h5'")