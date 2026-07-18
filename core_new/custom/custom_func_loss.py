import tensorflow as tf

def custom_precision_loss(y_true, y_pred, alpha=0.5):
    # alpha(гиперпараметр):
    # Контролирует влияние FP на общую потерю.Чем выше alpha, тем сильнее модель
    # избегает ложных срабатываний.
    # Если precision растет, но recall падает, уменьшите alpha или
    # добавьте штраф за ложноотрицательные (FN) через beta * (y_true * (1 - y_pred)).
    # Стандартная кросс-энтропия
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Штраф за ложноположительные (FP)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # FP = (1 - y_true) * y_pred
    fp = tf.reduce_sum((1 - y_true) * y_pred)

    # Комбинированная потеря: BCE + alpha * FP
    total_loss = bce + alpha * fp

    return total_loss