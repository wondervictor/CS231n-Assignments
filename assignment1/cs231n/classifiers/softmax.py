import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  D, C = W.shape
  N, _ = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  y_ = np.dot(X, W)

  for idx in xrange(N):
    y_softmax = np.exp(y_[idx]) / np.sum(np.exp(y_[idx]))
    loss -= np.log2(y_softmax[y[idx]])
    # dW[:, ] += y_softmax
    # dW[y[idx]] -= 1.0
    for j in xrange(C):
      dW[:, j] += X[idx] * y_softmax[j]
    dW[:, y[idx]] -= X[idx]

    # dW += np.reshape(X[idx], (D, 1)) * np.reshape(grad, (1, C))

  loss += 0.5 * reg * np.sum(W ** 2)

  dW /= N
  loss /= N
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  D, C = W.shape
  N, _ = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  y_ = np.dot(X, W)

  y_softmax = np.divide(np.exp(y_).T, np.sum(np.exp(y_), axis=1)).T
  y = y.tolist()
  loss = -np.log2(y_softmax[range(N), y])
  grad = y_softmax
  grad[range(N), y] -= 1.0
  loss = np.mean(loss)
  dW = np.reshape(X, (N, D, 1)) * np.reshape(grad, (N, 1, C)) #np.dot(X, grad)
  loss += 0.5 * reg * np.sum(W ** 2)/N
  dW = np.mean(dW, 0)
  dW += reg * W / N

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

