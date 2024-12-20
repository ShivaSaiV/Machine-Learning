import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, preprocess_data, split_data, accuracy, standardize

class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        """
        Initialize the perceptron.

        >>> p = Perceptron(learning_rate=0.1, max_epochs=100)
        >>> p.learning_rate
        0.1
        >>> p.max_epochs
        100
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None

    def activate(self, x):
        """
        Activation function for the perceptron.
        
        >>> p = Perceptron()
        >>> p.activate(1.5)
        1
        >>> p.activate(-0.5)
        -1
        """
        # >>> YOUR CODE HERE >>>
        if x >= 0:
            act = 1
        else:
            act = -1
        # <<< END OF YOUR CODE <<<
        return act

    def predict(self, X):
        """
        Predict the class labels for the input samples.
        
        >>> p = Perceptron()
        >>> p.weights = np.array([0.1, -0.2, 0.3])
        >>> X = np.array([[1, 2, 3], [1, 10, 6]])
        >>> p.predict(X)
        array([ 1, -1])
        """
        # >>> YOUR CODE HERE >>>
        prod = np.dot(X, self.weights)
        pred = []
        n = len(prod)
        for i in range(n):
            pred.append(self.activate(prod[i]))
        pred = np.array(pred)
        # <<< END OF YOUR CODE <<<
        return pred
    
    def train_one_epoch(self, X_train, y_train, X_val, y_val):
        """
        Train the perceptron model for one epoch.
        
        >>> p = Perceptron(learning_rate=0.1, max_epochs=1)
        >>> X_train = np.array([[1, 2], [3, 4]])
        >>> y_train = np.array([1, -1])
        >>> X_val = np.array([[5, 6]])
        >>> y_val = np.array([1])
        >>> p.weights = np.array([0.1, -0.1])
        >>> train_acc, val_acc = p.train_one_epoch(X_train, y_train, X_val, y_val)
        >>> 0 <= train_acc <= 1 and 0 <= val_acc <= 1
        True
        >>> np.allclose(p.weights, np.array([-0.1, -0.3]))
        True
        """
        train_acc = -1
        val_acc = -1
        # >>> YOUR CODE HERE >>>

        n_train = len(y_train)
        preds_train = self.predict(X_train)
        for i in range(n_train):
            pred = self.predict(X_train[i].reshape(1, -1))
            if pred == y_train[i]:
                continue
            else:
                self.weights += self.learning_rate * y_train[i] * X_train[i]
        train_acc = accuracy(y_train, preds_train)

        preds_val = self.predict(X_val)
        val_acc = accuracy(y_val, preds_val)

        # <<< END OF YOUR CODE <<<
        return train_acc, val_acc

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the perceptron model.
        
        >>> p = Perceptron(learning_rate=0.1, max_epochs=2)
        >>> X_train = np.array([[1, 2], [3, 4]])
        >>> y_train = np.array([1, -1])
        >>> X_val = np.array([[5, 6]])
        >>> y_val = np.array([1])
        >>> train_acc, val_acc = p.train(X_train, y_train, X_val, y_val)
        >>> len(train_acc) == len(val_acc) == 2
        True
        >>> np.allclose(train_acc, [0.5, 0.5])
        True
        """
        np.random.seed(42)
        n_samples, n_features = X_train.shape
        # Initialize the weights
        self.weights = np.random.normal(0, 1, n_features)
        train_accuracies = []
        val_accuracies = []

        # >>> YOUR CODE HERE >>>
        for i in range(self.max_epochs):
            train_acc, val_acc = self.train_one_epoch(X_train, y_train, X_val, y_val)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
        # <<< END OF YOUR CODE <<<

        return train_accuracies, val_accuracies

    def plot_learning_curve(self, train_accuracies, val_accuracies):
        """
        Plot the learning curve.
        """
        # >>> YOUR CODE HERE >>>
        plt.plot(train_accuracies, label = "Training")
        plt.plot(val_accuracies, label = "Validation")
        plt.title("Perceptron Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("learning_curve_perceptron.png")
        plt.close()
        # <<< END OF YOUR CODE <<<

def main():
    # Load and preprocess data
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    df = load_data(url)
    X, y = preprocess_data(df)
    y[y == 0] = -1  # set the negtive label
    X_train, X_test, y_train, y_test = split_data(X, y)
    # process the data
    X_train, X_test = standardize(X_train, X_test)
    
    # Train perceptron
    perceptron = Perceptron(learning_rate=0.01, max_epochs=100)
    train_accuracies, val_accuracies = perceptron.train(X_train, y_train, X_test, y_test)

    # Plot learning curve
    perceptron.plot_learning_curve(train_accuracies, val_accuracies)

    # Final accuracy
    final_train_acc = train_accuracies[-1]
    final_val_acc = val_accuracies[-1]
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()