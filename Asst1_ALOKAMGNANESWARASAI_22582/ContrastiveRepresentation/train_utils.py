import torch
from argparse import Namespace
from typing import Union, Tuple, List
import ContrastiveRepresentation.pytorch_utils as ptu
from utils import get_data_batch, get_contrastive_data_batch
from LogisticRegression.model import LinearModel
from LogisticRegression.train_utils import fit_model as fit_linear_model,\
    calculate_loss as calculate_linear_loss,\
    calculate_accuracy as calculate_linear_accuracy
import numpy as np


def calculate_loss(
        y_logits: torch.Tensor, y: torch.Tensor
) -> float:
    '''
    Calculate the loss of the model on the given data.

    Args:
        y_logits: torch.Tensor, softmax logits
        y: torch.Tensor, labels
    
    Returns:
        loss: float, loss of the model
    '''
    # raise NotImplementedError('Calculate negative-log-likelihood loss here')
    loss = torch.nn.CrossEntropyLoss()(y_logits, y)
    return loss.item()


def calculate_accuracy(
        y_logits: torch.Tensor, y: torch.Tensor
) -> float:
    '''
    Calculate the accuracy of the model on the given data.

    Args:
        Args:
        y_logits: torch.Tensor, softmax logits
        y: torch.Tensor, labels
    
    Returns:
        acc: float, accuracy of the model
    '''
    _, y_preds = torch.max(y_logits, 1)
    acc = (y_preds == y).float().mean()
    return acc.item()



def fit_contrastive_model(
        encoder: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        num_iters: int = 1000,
        batch_size: int = 256,
        learning_rate: float = 1e-3
) -> None:
    '''
    Fit the contrastive model.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - X: torch.Tensor, features
    - y: torch.Tensor, labels
    - num_iters: int, number of iterations for training
    - batch_size: int, batch size for training

    Returns:
    - losses: List[float], list of losses at each iteration
    '''
    # TODO: define the optimizer for the encoder only
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    # TODO: define the loss function
    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    losses = []
    
    encoder.train()

    for i in range(num_iters):
        
        # anchor_samples, positive_samples, negative_samples = get_contrastive_data_batch(ptu.to_numpy(X), ptu.to_numpy(y), batch_size)
        data_generator = get_contrastive_data_batch(ptu.to_numpy(X), ptu.to_numpy(y), batch_size)
        anchor_samples, positive_samples, negative_samples = next(data_generator)
        anchor_samples = ptu.from_numpy(anchor_samples).float()
        positive_samples = ptu.from_numpy(positive_samples).float()
        negative_samples = ptu.from_numpy(negative_samples).float()
        
        anchor_embeddings = encoder(anchor_samples)
        positive_embeddings = encoder(positive_samples)
        negative_embeddings = encoder(negative_samples)
        optimizer.zero_grad()
        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if i % 10 == 0:
            print(f'Iteration: {i}, Loss: {loss.item()}')
    return losses

        # raise NotImplementedError('Write the contrastive training loop here')
    
def evaluate_model(
        encoder: torch.nn.Module,
        classifier: Union[LinearModel, torch.nn.Module],
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 256,
        is_linear: bool = False
) -> Tuple[float, float]:
    '''
    Evaluate the model on the given data.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - classifier: Union[LinearModel, torch.nn.Module], the classifier model
    - X: torch.Tensor, images
    - y: torch.Tensor, labels
    - batch_size: int, batch size for evaluation
    - is_linear: bool, whether the classifier is linear

    Returns:
    - loss: float, loss of the model
    - acc: float, accuracy of the model
    '''
    # raise NotImplementedError('Get the embeddings from the encoder and pass it to the classifier for evaluation')
    
    temp = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        X_batch = encoder(X_batch)
        temp.append(X_batch)
    
    X = torch.cat(temp)
   
    # X = encoder(X)
    if is_linear:
      
        X = ptu.to_numpy(X)
    
    y_preds = classifier(X)
    
    if is_linear:
        return calculate_linear_loss(y_preds, y), calculate_linear_accuracy(y_preds, y)
    
    # HINT: use calculate_loss and calculate_accuracy functions for NN classifier and calculate_linear_loss and calculate_linear_accuracy functions for linear (softmax) classifier
    
    return calculate_loss(y_preds, y), calculate_accuracy(y_preds, y)


def fit_model(
        encoder: torch.nn.Module,
        classifier: Union[LinearModel, torch.nn.Module],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        args: Namespace
) -> Tuple[List[float], List[float], List[float], List[float]]:
    '''
    Fit the model.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - classifier: Union[LinearModel, torch.nn.Module], the classifier model
    - X_train: torch.Tensor, training images
    - y_train: torch.Tensor, training labels
    - X_val: torch.Tensor, validation images
    - y_val: torch.Tensor, validation labels
    - args: Namespace, arguments for training

    Returns:
    - train_losses: List[float], list of training losses
    - train_accs: List[float], list of training accuracies
    - val_losses: List[float], list of validation losses
    - val_accs: List[float], list of validation accuracies
    '''
    if args.mode == 'fine_tune_linear':
        # raise NotImplementedError('Get the embeddings from the encoder and use already implemeted training method in softmax regression')
        
        
        # X_train = ptu.to_numpy(encoder(X_train))   # this is correct
        
        temp = []
        for i in range(0, len(X_train), args.batch_size):
            X_batch = X_train[i:i+args.batch_size]
            X_batch = encoder(X_batch)
            X_batch = ptu.to_numpy(X_batch)
            temp.append(X_batch)
            
        X_train = np.concatenate(temp)
       
        X_val =  ptu.to_numpy(encoder(X_val))
        y_train = ptu.to_numpy(y_train)
        y_val = ptu.to_numpy(y_val)
        
        train_losses, train_accs, val_losses, val_accs = fit_linear_model(
            classifier, X_train, y_train, X_val, y_val, args.num_iters, args.lr, args.batch_size, args.l2_lambda, args.grad_norm_clip
        )
        
        return train_losses, train_accs, val_losses, val_accs
    
    
    else:
        # TODO: define the optimizer
        # raise NotImplementedError('Write the supervised training loop here')
        encoder.train()
        classifier.train()
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        criterion = torch.nn.CrossEntropyLoss()
        
        
        for i in range(args.num_iters):
            X_batch, y_batch = get_data_batch(X_train, y_train, args.batch_size)
            X= encoder(X_batch)
            y_preds = classifier(X)
            optimizer.zero_grad()
            loss = criterion(y_preds, y_batch)  
            acc = calculate_accuracy(y_preds, y_batch)
            loss.backward()
            optimizer.step()
           
            if i % 10== 0:
                train_losses.append(loss.item())
                train_accs.append(acc)
                val_loss, val_acc = evaluate_model(encoder, classifier, X_val, y_val, args.batch_size, False)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
            
                print(f'Iteration: {i}/{args.num_iters}, Loss: {loss.item()}, Acc: {acc}, Val Loss: {val_loss}, Val Acc: {val_acc}')
                
        # return the losses and accuracies both on training and validation data
        return train_losses, train_accs, val_losses, val_accs
