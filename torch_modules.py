import torch
import torch.nn as nn
import os, json

#Stem/Head Conv
def TorchConvBnAct(config_key, act='ReLU6', bn_momentum=0.1):
    in_c = config_key[0]
    out_c = config_key[1]
    k = config_key[2]
    s = config_key[3]
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, padding=k//2, bias=False),
        nn.BatchNorm2d(out_c, momentum=bn_momentum),
        nn.__dict__[act](inplace=True))

#Inverted Residual (MBConv) Block
class TorchMBConvBlock(nn.Module):
    def __init__(self, config_key, act='ReLU6', bn_momentum=0.1):
        super(TorchMBConvBlock, self).__init__()
        in_s = config_key[0]
        k = config_key[1]
        s = config_key[2]
        in_c = config_key[3]
        exp_c = config_key[4]
        out_c = config_key[5]

        self.residual_connection = (in_c == out_c and s == 1)
        self.expansion = (in_c != exp_c)
        if self.expansion:
            self.pw1 = nn.Sequential(
                    nn.Conv2d(in_c, exp_c, 1, bias=False),
                    nn.BatchNorm2d(exp_c, momentum=bn_momentum),
                    nn.__dict__[act](inplace=True))
        self.dw = nn.Sequential(
                nn.Conv2d(exp_c, exp_c, k, padding=k//2,
                          stride=s, groups=exp_c, bias=False),
                nn.BatchNorm2d(exp_c, momentum=bn_momentum),
                nn.__dict__[act](inplace=True))
        self.pw2 = nn.Sequential(
                nn.Conv2d(exp_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c, momentum=bn_momentum))

    def forward(self, input):
        if self.expansion:
            x = self.pw1(input)
        else:
            x = input
        x = self.dw(x)
        x = self.pw2(x)
        if self.residual_connection:
            return x + input
        else:
            return x

#MBConv Stage
class TorchMBConvStage(nn.Module):
    def __init__(self, block_configs: list, act='ReLU6', bn_momentum=0.1):
        super(TorchMBConvStage, self).__init__()
        assert type(block_configs[0]) is list

        blocks = []
        for config in block_configs:
            blocks.append(TorchMBConvBlock(config, act, bn_momentum))
        
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x
        
#MBConv Branched Stage
class TorchBranchedStage(nn.Module):
    def __init__(self, branch_configs, is_synced, sync_type, act='ReLU6', bn_momentum=0.1):
        super(TorchBranchedStage, self).__init__()

        self.is_synced = is_synced
        self.sync_type = sync_type

        branch1 = []
        branch2 = []
        for i in range(len(branch_configs['0'])):
            branch1.append(TorchMBConvBlock(branch_configs['0'][i], act, bn_momentum))
            branch2.append(TorchMBConvBlock(branch_configs['1'][i], act, bn_momentum))
        
        self.branch1 = nn.Sequential(*branch1)
        self.branch2 = nn.Sequential(*branch2)

    def _forward(self, x):
        if self.is_synced:
            branch_in1 = x
            branch_in2 = x
        else:
            branch_in1 = x[0]
            branch_in2 = x[1]
        
        branch_out1 = self.branch1(branch_in1)
        branch_out2 = self.branch2(branch_in2)

        if self.sync_type == 'n':
            return branch_out1, branch_out2
        elif self.sync_type == 'a':
            return torch.add(branch_out1, branch_out2)
        elif self.sync_type == 'c':
            return torch.cat((branch_out1, branch_out2), dim=1)

    def forward(self, x):
        x = self._forward(x)
        return x

#Build Model with Configs.json
class TorchPretrainedModel(nn.Module):
    def __init__(self, path, num_classes=1000, input_size=224, dropout=0.2):
        super(TorchPretrainedModel, self).__init__()

        assert os.path.exists(path)
        with open(path, 'r') as f:
            model_config = json.loads(f.read())

        self.name = model_config['name']
        self.act = model_config['act']
        self.bn_momentum = model_config['bn_momentum']
        self.last_c = model_config['head'][1]

        #Stem Conv
        self.stem_conv = TorchConvBnAct(model_config['stem'], self.act, self.bn_momentum)

        #MBConv Stages(1~7)
        stages = []
        for stage in range(7):
            stages.append(TorchMBConvStage(model_config[str(stage)], self.act, self.bn_momentum))
        self.stages = nn.Sequential(*stages)

        #Head Conv, Classifier
        self.head_conv = TorchConvBnAct(model_config['head'], self.act, self.bn_momentum)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(self.last_c, num_classes, bias=True))
        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stages(x)
        x = self.head_conv(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                                         nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)



#Build Model with Configs.json
class TorchBranchedModel(nn.Module):
    def __init__(self, path, num_classes=1000, input_size=224, dropout=0.2):
        super(TorchBranchedModel, self).__init__()

        assert os.path.exists(path)
        with open(path, 'r') as f:
            model_config = json.loads(f.read())

        self.name = model_config['name']
        self.act = model_config['act']
        self.bn_momentum = model_config['bn_momentum']
        self.last_c = model_config['head'][1]
        self.sync_type = model_config['sync']

        #Stem Conv
        self.stem_conv = TorchConvBnAct(model_config['stem'], self.act, self.bn_momentum)

        #MBConv Stages(1~7)/Branches
        self.stage1 = TorchBranchedStage(model_config['0'], self.is_synced(0), self.sync_type[0],
                                          self.act, self.bn_momentum)
        self.stage2 = TorchBranchedStage(model_config['1'], self.is_synced(1), self.sync_type[1],
                                          self.act, self.bn_momentum)
        self.stage3 = TorchBranchedStage(model_config['2'], self.is_synced(2), self.sync_type[2],
                                          self.act, self.bn_momentum)
        self.stage4 = TorchBranchedStage(model_config['3'], self.is_synced(3), self.sync_type[3],
                                          self.act, self.bn_momentum)
        self.stage5 = TorchBranchedStage(model_config['4'], self.is_synced(4), self.sync_type[4],
                                          self.act, self.bn_momentum)
        self.stage6 = TorchBranchedStage(model_config['5'], self.is_synced(5), self.sync_type[5],
                                          self.act, self.bn_momentum)
        self.stage7 = TorchBranchedStage(model_config['6'], self.is_synced(6), self.sync_type[6],
                                          self.act, self.bn_momentum)

        #Head Conv, Classifier
        self.head_conv = TorchConvBnAct(model_config['head'], self.act, self.bn_momentum)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(self.last_c, num_classes, bias=True))
        self._initialize_weights()

    def is_synced(self, stage):
        if stage != 0 and self.sync_type[stage-1] == 'n':
            return False
        else:
            return True

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.head_conv(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                                         nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)
