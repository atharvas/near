import os
import numpy as np

if __name__ == "__main__":
    train_data = 'data/ecg/x_train.npy'
    train_labels = 'data/ecg/y_train.npy'
    test_data = 'data/ecg/x_test.npy'
    test_labels = 'data/ecg/y_test.npy'
    input_type = 'list'
    output_type = 'atom'
    for arr_pth in [train_data, train_labels, test_data, test_labels]:
        arr = np.load(arr_pth)
        print(arr_pth)
        print(arr.shape)
        print(np.unique(arr))

        if os.path.basename(arr_pth).startswith('x'):
            # output shape: (num_samples, 1, num_features)
            assert input_type == 'list'
            flattened_arr = arr.reshape(arr.shape[0], 1, arr.shape[1])
        elif os.path.basename(arr_pth).startswith('y'):
            # output shape: (num_samples, num_features)
            assert output_type == 'atom'
            flattened_arr = arr.reshape(arr.shape[0], arr.shape[1])
        else:
            raise NotImplementedError(f"not train or not test")
   
        with open(arr_pth.replace('.npy', '_flat.npy'), 'wb') as f:
            np.save(f, flattened_arr)
