from .spine import SpineDataProvider, SpineAugmentedDataProvider


def get_data_provider_by_name(name, train_params):
    """Return required data provider class"""
    if name == 'SPINE':
        return SpineDataProvider(**train_params)
    if name == 'SPINE+':
        return SpineAugmentedDataProvider(**train_params)
    else:
        print("Sorry, data provider for `%s` dataset "
              "was not implemented yet" % name)
        exit()
