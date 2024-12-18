import numpy as np

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        """
        Выполняет fit модели LDA.
        Args:
        ----------
        X : numpy.ndarray
            Матрица признаков размерностью (n_samples, n_features).
        y : numpy.ndarray
            Вектор меток классов размерностью (n_samples,).

        Returns: self
        -----------
        """

        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within class scatter matrix
        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot(X_c - mean_c)

        # Between class scatter matrix
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(-1, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        # Solve the generalized eigenvalue problem for the matrix SW^-1 SB
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(SW).dot(SB))

        # Get the eigenvectors that correspond to the n_components largest eigenvalues
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        self.linear_discriminants = eigenvectors[:, :self.n_components]
        return self

    def transform(self, X):
        return np.dot(X, self.linear_discriminants)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)