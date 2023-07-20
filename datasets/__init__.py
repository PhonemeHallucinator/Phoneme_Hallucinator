import os
import pickle

def get_dataset(args, split):
    if args.dataset == 'speech':
        from .speech import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size, args.mask_type)
    else:
        raise ValueError()

    return dataset
    
def cache(args, split, fname):
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            batches = pickle.load(f)
    else:
        batches = []
        dataset = get_dataset(args, split)
        dataset.initialize()
        for _ in range(dataset.num_batches):
            batch = dataset.next_batch()
            batches.append(batch)
        with open(fname, 'wb') as f:
            pickle.dump(batches, f)

    return batches
