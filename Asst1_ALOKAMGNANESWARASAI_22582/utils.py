import torch
import numpy as np
from typing import Tuple
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

def get_data(
        data_path: str = 'data/cifar10_train.npz', is_linear: bool = False,
        is_binary: bool = False, grayscale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load CIFAR-10 dataset from the given path and return the images and labels.
    If is_linear is True, the images are reshaped to 1D array.
    If grayscale is True, the images are converted to grayscale.

    Args:
    - data_path: string, path to the dataset
    - is_linear: bool, whether to reshape the images to 1D array
    - is_binary: bool, whether to convert the labels to binary
    - grayscale: bool, whether to convert the images to grayscale

    Returns:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    '''
    data = np.load(data_path)
    X = data['images']
    try:
        y = data['labels']
    except KeyError:
        y = None
     
    X = X.transpose(0, 3, 1, 2)
    if is_binary:
        idxs0 = np.where(y == 0)[0]
        idxs1 = np.where(y == 1)[0]
        idxs = np.concatenate([idxs0, idxs1])
        X = X[idxs]
        y = y[idxs]
    if grayscale:
        
        X = convert_to_grayscale(X)
        
    if is_linear:
        X = X.reshape(X.shape[0], -1)
    
    # HINT: rescale the images for better (and more stable) learning and performance 
    # use standard scaler from sklearn.preprocessing
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
    
    
   
    
    return X, y


def convert_to_grayscale(X: np.ndarray) -> np.ndarray:
    '''
    Convert the given images to grayscale.

    Args:
    - X: np.ndarray, images in RGB format

    Returns:
    - X: np.ndarray, grayscale images
    '''
    X_gray = np.dot(X.transpose(0, 2, 3, 1)[..., :3], [0.2989, 0.5870, 0.1140])
    return np.expand_dims(X_gray, axis=1)
    # return np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])


def train_test_split(
        X: np.ndarray, y: np.ndarray, test_ratio: int = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split the given dataset into training and test sets.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - test_ratio: float, ratio of the test set

    Returns:
    - X_train: np.ndarray, training images
    - y_train: np.ndarray, training labels
    - X_test: np.ndarray, test images
    - y_test: np.ndarray, test labels
    '''
    assert test_ratio < 1 and test_ratio > 0
    
    noof_test_samples = int(test_ratio * X.shape[0])
    ind = np.random.permutation(X.shape[0]) 
    
    train_ind, test_ind = ind[noof_test_samples:], ind[:noof_test_samples]

    
    X_train, X_test = X[train_ind], X[test_ind]
    y_train, y_test = y[train_ind], y[test_ind]
    
    # raise NotImplementedError('Split the dataset here')
    
    return X_train, y_train, X_test, y_test


def get_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get a batch of the given dataset.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Returns:
    - X_batch: np.ndarray, batch of images
    - y_batch: np.ndarray, batch of labels
    '''
    idxs = np.random.choice(X.shape[0], size=batch_size, replace=False) # : get random indices of the batch size without replacement from the dataset
    return X[idxs], y[idxs]


# TODO: Read up on generator functions online
def get_contrastive_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
):  # Yields: Tuple[np.ndarray, np.ndarray, np.ndarray]
    '''
    Get a batch of the given dataset for contrastive learning.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Yields:
    - X_a: np.ndarray, batch of anchor samples
    - X_p: np.ndarray, batch of positive samples
    - X_n: np.ndarray, batch of negative samples
    '''
   
    
    while True:
        anchor_indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
        anchor_samples = X[anchor_indices]
        positive_indices = [np.random.choice(np.where(y == y[i])[0]) for i in anchor_indices]
        positive_samples = X[positive_indices]
        negative_indices = [np.random.choice(np.where(y != y[i])[0]) for i in anchor_indices]
        negative_samples = X[negative_indices]
        yield anchor_samples, positive_samples, negative_samples


def plot_losses(
        train_losses: list, val_losses: list, title: str
) -> None:
    '''
    Plot the training and validation losses.

    Args:
    - train_losses: list, training losses
    - val_losses: list, validation losses
    - title: str, title of the plot
    '''
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    if title == 'softmax - Losses':
        plt.savefig('images/1.1a.png')
        
    elif title == 'cont_rep - Losses':
        plt.savefig('images/2.2.png')
        
    elif title == 'fine_tune_nn - Losses':
        plt.savefig('images/2.5.png')
    
        
    elif title == 'fine_tune_linear - Losses':
        plt.savefig('images/2.4.png')
    else :
        plt.savefig(f'images/loss.png')
    plt.close()


def plot_accuracies(
        train_accs: list, val_accs: list, title: str
) -> None:
    '''
    Plot the training and validation accuracies.

    Args:
    - train_accs: list, training accuracies
    - val_accs: list, validation accuracies
    - title: str, title of the plot
    '''
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    
    if title == 'softmax - Accuracies':
        plt.savefig('images/1.1b.png')
        
    elif title == 'fine_tune_linear - Accuracies':
        plt.savefig('images/2.4b.png')
        
    else :
        plt.savefig(f'images/acc.png')
    
    plt.close()


def plot_tsne(
       z: torch.Tensor, y: torch.Tensor 
) -> None:
    '''
    Plot the 2D t-SNE of the given representation.

    Args:
    - z: torch.Tensor, representation
    - y: torch.Tensor, labels
    '''
    # z2 = # TODO: get 2D t-SNE of the representation
    tsne = TSNE(n_components=2, random_state=42)
    z2 = tsne.fit_transform(z)
    y = y.detach().cpu().numpy()
    plt.figure(figsize=(20, 20))
    for i in range(10):
        idxs = np.where(y == i)
        plt.scatter(z2[idxs, 0], z2[idxs, 1], label=str(i))
        
    plt.title('t-SNE of the Representation')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig('images/1.3.png')
    print('Saved t-SNE plot')
    plt.close()
