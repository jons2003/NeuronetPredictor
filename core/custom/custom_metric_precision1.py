import tensorflow as tf
from tensorflow.keras import backend as K


class PrecisionClass1(tf.keras.metrics.Metric):
    def __init__(self, name='precision_class1', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # Более стабильный расчет TP и FP
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)

    def result(self):
        # Добавление эпсилон для численной стабильности
        return self.tp / (self.tp + self.fp + K.epsilon())

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)


