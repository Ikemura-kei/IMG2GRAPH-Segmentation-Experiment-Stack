def init_params_lr(net, opt):
    bias_params = []
    nonbias_params = []
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
            else:
                nonbias_params.append(value)
    params = [
        {'params': nonbias_params,
         'lr': opt.lr,
         'weight_decay': opt.weight_decay},
        {'params': bias_params,
         'lr': opt.lr * 2.0,
         'weight_decay': 0}
    ]
    return params