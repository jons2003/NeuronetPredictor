import tensorflow as tf
from tensorflow.keras import backend as K


def focal_precision_loss(focal_param:dict):
    ###
    # gamma=2.0: Управляет фокусировкой на сложных примерах. Чем выше, вероятность ошибочного предскахания,тем больше штраф
    # alpha=0.8: Вес класса 1 (например, если класс 1 редкий, установите alpha > 0.5
    ###
    alpha = focal_param['alpha']
    gamma = focal_param['gamma']
    alpha_fp = focal_param['alpha_fp']

    def loss(y_true, y_pred):
        """
        Улучшенная версия функции потерь с:
        - Автоматическим расчетом весов классов
        - Балансировкой компонент потерь
        - Численной стабильностью
        """
        # Приведение типов и клиппинг значений
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        # Расчет базовой кросс-энтропии
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        print('bce  ',bce)
        # Расчет p_t с защитой от нулевых значений
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        print('p_t  ',p_t)
        p_t = tf.clip_by_value(p_t, K.epsilon(), 1.0 - K.epsilon())
        print('p_t  ',p_t)
        # Focal Loss компонента
        focal_loss = alpha * tf.pow(1.0 - p_t, gamma) * bce
        print('focal_loss   ',focal_loss)
        # Штраф за ложноположительные срезы (нормировка на размер батча)
        fp_loss = alpha_fp * tf.reduce_mean((1 - y_true) * y_pred)
        print('fp_loss   ',fp_loss)
        # Суммарные потери
        total_loss = tf.reduce_mean(focal_loss) + fp_loss
        return total_loss
    return  loss


# # Пример использования в модели
# def build_model(input_shape):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
#
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#         loss=focal_precision_loss,
#         metrics=[
#             PrecisionClass1(),
#             tf.keras.metrics.Recall(name='recall_1'),
#             tf.keras.metrics.AUC(name='auc')
#         ]
#     )
#     return model