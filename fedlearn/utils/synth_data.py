from FACT.helper import *
from FACT.fairness import *
from FACT.data_util import *
from FACT.plot import *
from FACT.lin_opt import *
from sklearn.utils import shuffle


def shuffle_clients_data(clients_data, p):
    def shuffle_arrays(arrays, indices):
        """Shuffles multiple arrays with the same indices."""
        return [array[indices] for array in arrays]

    # Get data from clients
    client_0_data = clients_data['client_0']
    client_1_data = clients_data['client_1']

    # Determine the number of data points to shuffle
    num_data_points_0_train = len(client_0_data['X_train'])
    num_data_points_1_train = len(client_1_data['X_train'])
    num_data_points_0_test = len(client_0_data['X_test'])
    num_data_points_1_test = len(client_1_data['X_test'])

    num_to_shuffle_train = int(min(num_data_points_0_train, num_data_points_1_train) * p)
    num_to_shuffle_test = int(min(num_data_points_0_test, num_data_points_1_test) * p)

    # Generate shuffle indices for both clients
    shuffle_indices_0_train = np.random.permutation(num_data_points_0_train)[:num_to_shuffle_train]
    shuffle_indices_1_train = np.random.permutation(num_data_points_1_train)[:num_to_shuffle_train]
    shuffle_indices_0_test = np.random.permutation(num_data_points_0_test)[:num_to_shuffle_test]
    shuffle_indices_1_test = np.random.permutation(num_data_points_1_test)[:num_to_shuffle_test]

    # Extract data to be shuffled from both clients for training and test datasets
    keys_train = ['X_train', 'y_train', 'X_train_removed']
    keys_test = ['X_test', 'y_test', 'X_test_removed']

    data_0_to_shuffle_train = [client_0_data[key][shuffle_indices_0_train] for key in keys_train]
    data_1_to_shuffle_train = [client_1_data[key][shuffle_indices_1_train] for key in keys_train]
    data_0_to_shuffle_test = [client_0_data[key][shuffle_indices_0_test] for key in keys_test]
    data_1_to_shuffle_test = [client_1_data[key][shuffle_indices_1_test] for key in keys_test]

    # Swap data between clients
    for i, key in enumerate(keys_train):
        client_0_data[key][shuffle_indices_0_train] = data_1_to_shuffle_train[i]
        client_1_data[key][shuffle_indices_1_train] = data_0_to_shuffle_train[i]

    for i, key in enumerate(keys_test):
        client_0_data[key][shuffle_indices_0_test] = data_1_to_shuffle_test[i]
        client_1_data[key][shuffle_indices_1_test] = data_0_to_shuffle_test[i]

    return clients_data

def split_client_data(X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, client_no=2, sex_idx = 2, split_method='iid'):

    

    if split_method=='iid':
        clients_data = {}


        X_train, y_train, X_train_removed = shuffle(X_train, y_train, X_train_removed, random_state=42)
        X_test, y_test, X_test_removed = shuffle(X_test, y_test, X_test_removed, random_state=42)
        
        # Split the data across clients
        X_train = np.split(X_train, client_no)
        y_train = np.split(y_train, client_no)
        X_test = np.split(X_test, client_no)
        y_test = np.split(y_test, client_no)
        X_train_removed = np.split(X_train_removed, client_no)
        X_test_removed = np.split(X_test_removed, client_no)

        for i in range(client_no):
            client_key = f"client_{i}"
            client_data = {'X_train': X_train[i], 'y_train': y_train[i],'X_test': X_test[i], 'y_test': y_test[i],'X_train_removed': X_train_removed[i], 'X_test_removed': X_test_removed[i] }
            clients_data[client_key] = client_data

    if split_method=='heterogeneity':
        clients_data = {} # alpha = 0.85

        # Separate data based on the sensitive attribute
        male_indices_train = X_train[:, sex_idx] == 1
        female_indices_train = X_train[:, sex_idx] == 0

        male_indices_test = X_test[:, sex_idx] == 1
        female_indices_test = X_test[:, sex_idx] == 0

        # Get data points corresponding to males and females
        X_train_male, y_train_male = X_train[male_indices_train], y_train[male_indices_train]
        X_train_female, y_train_female = X_train[female_indices_train], y_train[female_indices_train]

        X_test_male, y_test_male = X_test[male_indices_test], y_test[male_indices_test]
        X_test_female, y_test_female = X_test[female_indices_test], y_test[female_indices_test]

        # Get 'removed' versions without the sensitive attribute
        X_train_removed_male = X_train_removed[male_indices_train]
        X_train_removed_female = X_train_removed[female_indices_train]
        X_test_removed_male = X_test_removed[male_indices_test]
        X_test_removed_female = X_test_removed[female_indices_test]

        # Shuffle data
        X_train_male, y_train_male, X_train_removed_male = shuffle(X_train_male, y_train_male, X_train_removed_male, random_state=42)
        X_train_female, y_train_female, X_train_removed_female = shuffle(X_train_female, y_train_female, X_train_removed_female, random_state=42)

        X_test_male, y_test_male, X_test_removed_male = shuffle(X_test_male, y_test_male, X_test_removed_male, random_state=42)
        X_test_female, y_test_female, X_test_removed_female = shuffle(X_test_female, y_test_female, X_test_removed_female, random_state=42)

        # Get 90% of male data for client 1 and 10% for client 2, and vice versa for female data
        split_ratio = 0.92
        clients_data['client_0'] = {
            'X_train': np.vstack((X_train_male[:int(split_ratio*len(X_train_male))], X_train_female[:int((1-split_ratio)*len(X_train_female))])),
            'y_train': np.concatenate((y_train_male[:int(split_ratio*len(y_train_male))], y_train_female[:int((1-split_ratio)*len(y_train_female))])),
            'X_test': np.vstack((X_test_male[:int(split_ratio*len(X_test_male))], X_test_female[:int((1-split_ratio)*len(X_test_female))])),
            'y_test': np.concatenate((y_test_male[:int(split_ratio*len(y_test_male))], y_test_female[:int((1-split_ratio)*len(y_test_female))])),
            'X_train_removed': np.vstack((X_train_removed_male[:int(split_ratio*len(X_train_removed_male))], X_train_removed_female[:int((1-split_ratio)*len(X_train_removed_female))])),
            'X_test_removed': np.vstack((X_test_removed_male[:int(split_ratio*len(X_test_removed_male))], X_test_removed_female[:int((1-split_ratio)*len(X_test_removed_female))])),
        }

        clients_data['client_1'] = {
            'X_train': np.vstack((X_train_male[int(split_ratio*len(X_train_male)):], X_train_female[int((1-split_ratio)*len(X_train_female)):])),
            'y_train': np.concatenate((y_train_male[int(split_ratio*len(y_train_male)):], y_train_female[int((1-split_ratio)*len(y_train_female)):])),
            'X_test': np.vstack((X_test_male[int(split_ratio*len(X_test_male)):], X_test_female[int((1-split_ratio)*len(X_test_female)):])),
            'y_test': np.concatenate((y_test_male[int(split_ratio*len(y_test_male)):], y_test_female[int((1-split_ratio)*len(y_test_female)):])),
            'X_train_removed': np.vstack((X_train_removed_male[int(split_ratio*len(X_train_removed_male)):], X_train_removed_female[int((1-split_ratio)*len(X_train_removed_female)):])),
            'X_test_removed': np.vstack((X_test_removed_male[int(split_ratio*len(X_test_removed_male)):], X_test_removed_female[int((1-split_ratio)*len(X_test_removed_female)):])),
        }

    if split_method=='synergy':
        clients_data = {}

        # Create masks for each condition
        condition_00 = (X_train[:, sex_idx] == 0) & (y_train == 0)
        condition_11 = (X_train[:, sex_idx] == 1) & (y_train == 1)
        condition_01 = (X_train[:, sex_idx] == 0) & (y_train == 1)
        condition_10 = (X_train[:, sex_idx] == 1) & (y_train == 0)

        # Separate data based on the conditions
        X_train_00, y_train_00 = X_train[condition_00], y_train[condition_00]
        X_train_11, y_train_11 = X_train[condition_11], y_train[condition_11]
        X_train_01, y_train_01 = X_train[condition_01], y_train[condition_01]
        X_train_10, y_train_10 = X_train[condition_10], y_train[condition_10]

        # Separate the 'removed' datasets in the same manner
        X_train_removed_00 = X_train_removed[condition_00]
        X_train_removed_11 = X_train_removed[condition_11]
        X_train_removed_01 = X_train_removed[condition_01]
        X_train_removed_10 = X_train_removed[condition_10]

        # Do the same for the test dataset
        condition_00_test = (X_test[:, sex_idx] == 0) & (y_test == 0)
        condition_11_test = (X_test[:, sex_idx] == 1) & (y_test == 1)
        condition_01_test = (X_test[:, sex_idx] == 0) & (y_test == 1)
        condition_10_test = (X_test[:, sex_idx] == 1) & (y_test == 0)

        X_test_00, y_test_00 = X_test[condition_00_test], y_test[condition_00_test]
        X_test_11, y_test_11 = X_test[condition_11_test], y_test[condition_11_test]
        X_test_01, y_test_01 = X_test[condition_01_test], y_test[condition_01_test]
        X_test_10, y_test_10 = X_test[condition_10_test], y_test[condition_10_test]

        # Separate the 'removed' test datasets in the same manner
        X_test_removed_00 = X_test_removed[condition_00_test]
        X_test_removed_11 = X_test_removed[condition_11_test]
        X_test_removed_01 = X_test_removed[condition_01_test]
        X_test_removed_10 = X_test_removed[condition_10_test]

        # Assign data to clients
        clients_data['client_0'] = {
            'X_train': np.vstack((X_train_00, X_train_11)),
            'y_train': np.concatenate((y_train_00, y_train_11)),
            'X_test': np.vstack((X_test_00, X_test_11)),
            'y_test': np.concatenate((y_test_00, y_test_11)),
            'X_train_removed': np.vstack((X_train_removed_00, X_train_removed_11)),
            'X_test_removed': np.vstack((X_test_removed_00, X_test_removed_11)),
        }

        clients_data['client_1'] = {
            'X_train': np.vstack((X_train_01, X_train_10)),
            'y_train': np.concatenate((y_train_01, y_train_10)),
            'X_test': np.vstack((X_test_01, X_test_10)),
            'y_test': np.concatenate((y_test_01, y_test_10)),
            'X_train_removed': np.vstack((X_train_removed_01, X_train_removed_10)),
            'X_test_removed': np.vstack((X_test_removed_01, X_test_removed_10)),
        }

    clients_data = shuffle_clients_data(clients_data, 0.01)
    return clients_data

def synth_process(data_num):
    X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, dtypes, dtypes_, sens_idc, race_idx, sex_idx = get_dataset('synth3', corr_sens=True, alpha = 0.7, data_num=data_num)
    print(X_train.shape)
    client_data = split_client_data(X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, client_no=2,sex_idx=2, split_method='synergy')
    return client_data
