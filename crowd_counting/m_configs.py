#-*-coding:utf-8-*-

class ConfigFactory:

    def __init__(self):
        self.model_name = "mcnn"
        self.batch_size = 1
        self.learing_rate = 0.000001
        self.learing_rate_decay = 0.9
        self.momentum = 0.9
        self.iter_num = 200000
        self.max_ckpt_keep = 2000
        self.ckpt_router = './ckpts/' + self.model_name
        self.log_router = './logs/' + self.model_name + r'/'

    def set_configs(self, model_name='mcnn', batch_size=1, learing_rate=0.00001, learing_rate_decay=0.9, momentum=0.9, iter_num=200000, max_ckpt_keep=2000):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learing_rate = learing_rate
        self.learing_rate_decay = learing_rate_decay
        self.momentum = momentum
        self.iter_num = iter_num
        self.max_ckpt_keep = max_ckpt_keep
        self.ckpt_router = './ckpts/' + self.model_name
        self.log_router = './logs/' + self.model_name + r'.log'

    def display_configs(self):
        msg = '''
        ------------ info of %s model -------------------
        batch size              : %s
        learing rate            : %f
        learing rate decay      : %f
        momentum                : %f
        iter num                : %s
        max ckpt keep           : %s
        ckpt router             : %s
        log router              : %s
        ------------------------------------------------
        ''' % (self.model_name, self.batch_size, self.learing_rate, self.learing_rate_decay, self.momentum, self.iter_num, self.max_ckpt_keep, self.ckpt_router, self.log_router)
        print(msg)
        return msg

if __name__ == '__main__':
    configs = ConfigFactory()
    configs.set_configs(model_name='mmmm')
    configs.display_configs()
