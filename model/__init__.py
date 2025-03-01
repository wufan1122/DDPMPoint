import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m


def create_pdec_model(opt):
    from.pdec_model import Pdec as M
    m = M(opt)
    logger.info('Pdec Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def create_des_model(opt):
    from.des_model import des as M
    m = M(opt)
    logger.info('Pdec Model [{:s}] is created.'.format(m.__class__.__name__))
    return m