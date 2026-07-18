import os
import yaml

class Env:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_name = config_file.split('/')[-1].split('.')[0]

    def __call__(self, key=None):
        if key:
            return  self.env_get_key(key)
        else: return os.environ

    def _flatten_dict(self, nested_dict, parent_key=''):
        flattened = {}
        for key, value in nested_dict.items():
            new_key = f"{key}" if parent_key else key
            if isinstance(value, dict):
                flattened.update(self._flatten_dict(value, new_key))
            else:
                flattened[new_key] = value
        return flattened


    def yaml_get(self):
        with open(self.config_file) as f:
            config = yaml.safe_load(f)
        env_dic = self._flatten_dict(config)
        focal_range = {}
        for param in ['alpha_range', 'gamma_range', 'alpha_fp_range']:
            focal_range.setdefault(param, list(map(float, env_dic[param].split(' '))))
        env_dic.update(focal_range)
        return env_dic

    def yaml_get_tuple(self, key):
        str_ = os.environ[self.model_name]
        dic = {}
        str_ = ''.join(s for s in str_ if s not in ['{', '}', '\'', ''])
        lst = list(map(lambda x: x.split(':'), str_.split(',')))
        lst = list(map(lambda x: (x[0].strip(' '), x[1].strip(' ')), lst))
        for el in lst:
            if key == el[0]:
                return list(map(float, el[-1].split(' ')))

    def env_set(self):
        with open( self.config_file) as f:
            config = yaml.safe_load(f)
        os.environ['config'] = str(self._flatten_dict(config))

    def env_get_key(self, key:str):
        str_ = os.environ[self.config_name]
        dic = {}
        str_ = ''.join(s for s in str_ if s not in ['{','}', '\'', ' '])
        lst = list(map(lambda x: x.split(':') , str_.split(',')))
        for el in lst:
            val = el[1]
            lst_tmp = []
            if len(el) > 2:
                for i in range(1,len(el)):
                    lst_tmp.append(el[i])
                    val = lst_tmp
            dic.setdefault(el[0], val)
        return dic[key]

    def env_del(self):
        del os.environ[self.model_name]

    def env_get_tuple(self, key):
        str_ = os.environ[self.model_name]
        dic = {}
        str_ = ''.join(s for s in str_ if s not in ['{','}', '\'', ''])
        lst = list(map(lambda x: x.split(':') , str_.split(',')))
        lst = list(map(lambda x: (x[0].strip(' '), x[1].strip(' ')),  lst))
        for el in lst:
            if key == el[0]:
                return list(map(float, el[-1].split(' ')))


