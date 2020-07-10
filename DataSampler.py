def DataSampler(validation_split,dataset_size):
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices,test_indices = indices[split:], indices[:split],indices[:split]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    return train_sampler,valid_sampler,test_sampler