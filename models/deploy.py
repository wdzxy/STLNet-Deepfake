import copy


def convert_to_deploy(model, inplace=False):
    model = model if inplace else copy.deepcopy(model)
    model.eval()
    model.switch_to_deploy()
    return model
