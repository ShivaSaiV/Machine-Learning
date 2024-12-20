import numpy as np
import random
import matplotlib.pyplot as plt
from utils import load_data, preprocess_data, split_data, accuracy, standardize

class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, max_epochs=1000, tolerance=1e-4, l2_penalty=0.01):
        """
        Initialize the logistic regression classifier with hyperparameters.
        
        >>> clf = LogisticRegressionClassifier(learning_rate=0.05, max_epochs=500)
        >>> clf.learning_rate
        0.05
        >>> clf.max_epochs
        500
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None

    def sigmoid(self, z) -> np.array:
        """
        Apply the sigmoid function.
        
        >>> clf = LogisticRegressionClassifier()
        >>> np.round(clf.sigmoid(np.array([0, 1, -1])), 4)
        array([0.5   , 0.7311, 0.2689])
        """
        # >>> YOUR CODE HERE >>>
        sig = 1 / (1 + (np.e ** (-z)))
        # <<< END OF YOUR CODE <<<
        return sig

    def predict_proba(self, X) -> np.array:
        """
        Predict the probability that each input belongs to the positive class.
        
        >>> X = np.array([[1, 2], [1, 3]])
        >>> clf = LogisticRegressionClassifier()
        >>> clf.weights = np.array([0.5, -0.5])
        >>> clf.bias = 0
        >>> np.round(clf.predict_proba(X), 4)
        array([0.3775, 0.2689])
        """
        # >>> YOUR CODE HERE >>>
        z = np.dot(X, self.weights) + self.bias
        proba = self.sigmoid(z)
        # <<< END OF YOUR CODE <<<
        return proba

    def predict(self, X) -> np.array:
        """
        Predict the binary class label for each input. Should be the type of int.
        
        >>> X = np.array([[1, 2], [1, 3]])
        >>> clf = LogisticRegressionClassifier()
        >>> clf.weights = np.array([0.5, -0.5])
        >>> clf.bias = 0
        >>> clf.predict(X)
        array([0, 0])
        """
        probabilities = self.predict_proba(X)
        # >>> YOUR CODE HERE >>>
        predictions = []
        for i in range(len(probabilities)):
            if probabilities[i] >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        predictions = np.array(predictions)
        # <<< END OF YOUR CODE <<<
        return predictions

    def base_logistic_loss(self, y_true, y_pred):
        """
        Compute the base logistic loss (without regularization).
        
        >>> y_true = np.array([0, 1])
        >>> y_pred = np.array([0.25, 0.75])
        >>> clf = LogisticRegressionClassifier()
        >>> np.round(clf.base_logistic_loss(y_true, y_pred), 4)
        0.2877
        """
        epsilon = 1e-15  # To avoid log(0)
        # >>> YOUR CODE HERE >>>
        base_loss = np.sum((y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)))
        base_loss *= -1 * 1/len(y_true)
        # <<< END OF YOUR CODE <<<
        return base_loss
    
    def l2_regularization_loss(self):
        """
        Compute the L2 regularization loss.
        
        >>> clf = LogisticRegressionClassifier(l2_penalty=0.1)
        >>> clf.weights = np.array([0.5, -0.5])
        >>> np.round(clf.l2_regularization_loss(), 4)
        0.025
        """
        # >>> YOUR CODE HERE >>>
        l2_loss = np.sum((self.weights ** 2)) 
        l2_loss *= self.l2_penalty/2
        # <<< END OF YOUR CODE <<<
        return l2_loss

    def logistic_loss(self, y_true, y_pred):
        """
        Compute the logistic loss (with L2 regularization).
        
        >>> y_true = np.array([0, 1])
        >>> y_pred = np.array([0.25, 0.75])
        >>> clf = LogisticRegressionClassifier(l2_penalty=0.1)
        >>> clf.weights = np.array([0.5, -0.5])
        >>> clf.bias = 0.0
        >>> np.round(clf.logistic_loss(y_true, y_pred), 4)
        0.3127
        """
        # >>> YOUR CODE HERE >>>
        loss = self.base_logistic_loss(y_true, y_pred) + self.l2_regularization_loss()
        # <<< END OF YOUR CODE <<<
        return loss

    def gradient(self, X, y):
        """
        Compute the gradient of the loss function with respect to weights and bias.
        
        >>> X = np.array([[1, 2], [1, 3]])
        >>> y = np.array([0, 1])
        >>> clf = LogisticRegressionClassifier()
        >>> clf.weights = np.array([0.5, -0.5])
        >>> clf.bias = 0
        >>> grad_w, grad_b = clf.gradient(X, y)
        >>> grad_w.shape
        (2,)
        >>> np.round(grad_b, 4)
        -0.1768
        """
        # >>> YOUR CODE HERE >>>
        y_hat = self.predict_proba(X)
        difference = y_hat - y
        gradient_w = np.dot(X.T, difference) + (self.l2_penalty * self.weights)
        gradient_w *= 1 / len(y)

        gradient_b = np.sum(difference)
        gradient_b *= 1 / len(y)
        # <<< END OF YOUR CODE <<<
        return gradient_w, gradient_b
    
    def train_one_epoch(self, X, y):
        """
        Perform one epoch of gradient descent to update the weights and bias.
        
        >>> X = np.array([[1, 2], [1, 3]])
        >>> y = np.array([0, 1])
        >>> clf = LogisticRegressionClassifier()
        >>> clf.weights = np.array([0.5, -0.5])
        >>> clf.bias = 0
        >>> clf.train_one_epoch(X, y)
        >>> np.round(clf.weights, 4)
        array([ 0.5017, -0.4928])
        >>> np.round(clf.bias, 4)
        0.0018
        """
        # >>> YOUR CODE HERE >>>
        gradient_w, gradient_b = self.gradient(X, y)
        self.weights = self.weights - self.learning_rate * gradient_w
        self.bias = self.bias - self.learning_rate * gradient_b
        # <<< END OF YOUR CODE <<<

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the logistic regression classifier on the training data.
        
        >>> X_train = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
        >>> y_train = np.array([0, 1, 1, 0])
        >>> X_val = np.array([[1, 2], [1, 3]])
        >>> y_val = np.array([0, 1])
        >>> clf = LogisticRegressionClassifier(learning_rate=0.01, max_epochs=2)
        >>> train_accs, test_accs, train_losses, test_losses = clf.train(X_train, y_train, X_val, y_val)
        >>> len(train_accs) == len(test_accs) == len(train_losses) == len(test_losses) == 2
        True
        >>> np.round(train_losses, 4)
        array([0.6975, 0.6975])
        """
        np.random.seed(42)
        random.seed(42)
        n_samples, n_features = X_train.shape
        self.weights = np.random.normal(0, 1, n_features)
        self.bias = 0.0

        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []

        loss = float("inf")

        for epoch in range(self.max_epochs):
            # >>> YOUR CODE HERE >>>
            self.train_one_epoch(X_train, y_train)

            pred_t = self.predict(X_train)
            preds_train = self.predict_proba(X_train)

            train_acc = accuracy(y_train, pred_t)
            train_accuracies.append(train_acc)
            
            pred_v = self.predict(X_val)
            preds_val = self.predict_proba(X_val)
            val_acc = accuracy(y_val, pred_v)
            val_accuracies.append(val_acc)

            train_loss = self.logistic_loss(y_train, preds_train)
            train_losses.append(train_loss)
            val_loss = self.logistic_loss(y_val, preds_val)
            val_losses.append(val_loss)

            if (abs(train_loss - loss) < self.tolerance):
                break
            
            loss = train_loss

            # <<< END OF YOUR CODE <<<

        return train_accuracies, val_accuracies, train_losses, val_losses

    def plot_logistic_regression_curve(self, train_accuracies, val_accuracies, train_losses, val_losses):
        """
        Plot the learning curve for loss and accuracy.
        """
        # >>> YOUR CODE HERE >>>
        plt.figure(figsize=(16, 10))
        plt.suptitle("Logistic Regression Learning Curve")

        plt.subplot(2, 1, 1)
        plt.plot(train_accuracies, label = "Train Accuracies")
        plt.plot(val_accuracies, label = "Validation Accuracies")
        plt.title("Accuracies Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(train_losses, label = "Training Losses")
        plt.plot(val_losses, label = "Validation Losses")
        plt.title("Validation Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Losses")
        plt.legend()

        plt.savefig("learning_curve_lr.png")
        plt.close()
        # <<< END OF YOUR CODE <<<

def main():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    df = load_data(url)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test = standardize(X_train, X_test)

    lr = LogisticRegressionClassifier(learning_rate=0.1, max_epochs=150, tolerance=1e-4)
    train_acc, val_acc, train_loss, val_loss = lr.train(X_train, y_train, X_test, y_test)

    lr.plot_logistic_regression_curve(train_acc, val_acc, train_loss, val_loss)

    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
    