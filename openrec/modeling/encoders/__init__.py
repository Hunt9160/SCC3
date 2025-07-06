__all__ = ['build_encoder']


def build_encoder(config):
    # from .rec_mobilenet_v3 import MobileNetV3
    from .scc3 import SCC3
    support_dict = ['SCC3']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'when encoder of rec model only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
