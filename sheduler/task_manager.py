from environment import Env
from core_new.get_data import Df_getter
from core_new.prepocessing import Preproccessing
from core_new.model import Model
from core_new.predict import Predicate
import os


class TaskManager:

    def __init__(self,  model_name:str, config_file='/home/neuro_linux/configs/config.yaml'):
        self.model_name = model_name
        self.env = Env(config_file=config_file)
        # self.env.env_set(model_name=model_name)  # Инициализация окружения из файла конфигурации

    def create_model(self, fit_params:dict, focal_params:dict):
        # fit_params = {'epochs': 200, 'batch_size': 64, 'class_weight': {0: 1, 1: 10}}
        # focal_params = {'alpha': float(env('alpha')), 'gamma': float(env('gamma')), 'alpha_fp': float(env('alpha_fp'))}

        predata = (Df_getter(self.env(key='data_db')).
                   mask(vol_data=self.env('vol_data'),
                        mask_name=self.env('mask_name'),
                        args={'data_synthetic': self.env('data_synthetic'),
                              'target_version': self.env('target_version')})
                   )
        predata = Preproccessing(predata=predata, preproc_file=self.env('models_dir') + self.model_name + '.proc',
                                 balance_method=self.env('balance_method')).learning()


        early_stopping_params = {
                                 # 'monitor': 'val_precision_class1',  # Мониторим точность валидационной выборки
                                 'monitor': 'val_keras_precision',
                                 'patience': 30,  # Количество эпох без улучшения перед остановкой
                                 'verbose': 1,
                                 'mode': 'max',  # Максимизируем точность класса 1
                                 'restore_best_weights': True  # Восстанавливаем веса лучшей модели
                                 }
        model_checkpoint_params = {'filepath': f"{self.env('models_dir')}{self.model_name}.keras",
                                   # 'monitor': 'val_precision_class1',  # Сохраняем модель с лучшим val_precision_class1
                                   'monitor': 'val_keras_precision','save_best_only': True,
                                   'save_weights_only': False,  # False для сохранения всей модели
                                   'mode': 'max',
                                   'verbose': 1}



        out_data = Model(data_proc=predata,
                         focal_params=focal_params,
                         fit_params=fit_params,
                         early_stopping_params=early_stopping_params,
                         model_checkpoint_params=model_checkpoint_params
                        ).learning()
        return out_data

    def predicate_model(self):
        self.env.env_set()

        predata = (Df_getter(self.env(key='prematch_db')).
                   mask(vol_data=self.env('vol_data'),
                        mask_name=self.env('mask_name'),
                        args={'data_synthetic': self.env('data_synthetic'),
                              'target_version': self.env('target_version')})
                       )

        predata = Preproccessing(predata=predata, preproc_file=self.env('models_dir') + self.model_name + '.proc').predicate()
        print(predata)
        (Predicate(predata=predata, model_path=self.env('models_dir') + self.model_name + '.keras').
         predicate({'alpha':1.5, 'gamma':3.0, 'alpha_fp':1.0}))


env = Env('/home/neuro_linux/configs/config.yaml')
env.env_set()
fit_params = {'epochs': 200, 'batch_size': 32, 'class_weight': {0: 1, 1: 1}}
focal_params = {'alpha': 7, 'gamma':5.0, 'alpha_fp': 50}
TaskManager('test').create_model(fit_params, focal_params)

TaskManager(model_name='model_216').predicate_model()

