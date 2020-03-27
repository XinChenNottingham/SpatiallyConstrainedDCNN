from models.model import Model

class MyModel(Model):

    def __init__(self, net):
        super().__init__(net)

    def get_grads(self, data_dict):
        pass

    def eval(self, data_dict, **kwargs):
        pass

    def predict(self, data_dict):
        pass