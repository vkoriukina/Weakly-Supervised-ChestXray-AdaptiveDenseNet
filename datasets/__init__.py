from . import chestxray14_dataset


def get_datasets(opts):
    if opts.dataset == 'ChestXray14':
        trainset = chestxray14_dataset.ChestXray14_Train(opts)
        valset = chestxray14_dataset.ChestXray14_Vali(opts)
        testset = chestxray14_dataset.ChestXray14_Test(opts)
    elif opts.test_list_box:
        testset = chestxray14_dataset.ChestXray14_Test(opts)
        return None, None, testset

    else:
        raise NotImplementedError

    return trainset, valset, testset
