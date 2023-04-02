import torch
from methods.base import Base_Client, Base_Server

class Client(Base_Client):
    # super의 init을 한번 실행하고 아래 코드도 실행
    # 부모 class의 함수는 모두 가지고 있음 
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        if 'NIH' in self.dir or 'CheXpert' in self.dir:
            self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        
class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes)
