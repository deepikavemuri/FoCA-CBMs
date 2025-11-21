# -*- coding: utf-8 -*- 
"""
Linear probing following example in https://github.com/openai/CLIP
Adapted from: https://github.com/KaiyangZhou/CoOp/blob/main/lpclip/linear_probe.py

@Time : 2024/10/17 16:18
@File :     scikitProbe.py 
"""
import random
import logging
import numpy as np

from tqdm import tqdm


class LogisticRegressionSearch:
    def __init__(self, n_runs, steps, n_jobs=-1, device='cpu'):
        super().__init__()
        self.n_runs = n_runs
        self.steps = steps
        self.val_acc_step_list = np.zeros([n_runs, steps])
        self.search_list = [1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6]
        self.acc_list = []
        self.best_c_weights_list = []
        self.c_left = None
        self.c_right = None
        self.n_jobs = n_jobs
        self.device = device

    def init_logistic_regression(self, C):
        if self.device == 'cpu':
            logging.info("Using Scikit Logistic Regression")
            from sklearn.linear_model import LogisticRegression
            probe = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, penalty="l2", n_jobs=self.n_jobs)
        else:
            logging.info("Using CuML Logistic Regression")
            from cuml.linear_model import LogisticRegression
            probe = LogisticRegression(C=C, solver="qn", max_iter=1000, penalty="l2")
        return ScikitLogisticRegression(probe=probe)

    def binary_search(self, train_data, val_data, seed, step):
        clf_left = self.init_logistic_regression(C=self.c_left)
        clf_left.train(*train_data)
        acc_left = clf_left.predict(*val_data)
        print("Val accuracy (Left): {:.2f}".format(acc_left), flush=True)

        clf_right = self.init_logistic_regression(C=self.c_right)
        clf_right.train(*train_data)
        acc_right = clf_right.predict(*val_data)
        print("Val accuracy (Right): {:.2f}".format(acc_right), flush=True)

        # find maximum and update ranges
        if acc_left < acc_right:
            acc_final = acc_right
            # range for the next step
            c_left = 0.5 * (np.log10(self.c_right) + np.log10(self.c_left))
            c_right = np.log10(self.c_right)
        else:
            acc_final = acc_left
            # range for the next step
            c_left = np.log10(self.c_left)
            c_right = 0.5 * (np.log10(self.c_right) + np.log10(self.c_left))
        self.c_left = np.power(10, c_left)
        self.c_right = np.power(10, c_right)
        logging.info(f"Val Accuracy: {acc_final:.2f}")
        self.val_acc_step_list[seed - 1, step] = acc_final

    def search(self, train_data, val_data, test_data):
        for seed in range(1, self.n_runs + 1):
            np.random.seed(seed)
            random.seed(seed)
            for C in tqdm(self.search_list, desc=f"List Searching (Seed {seed} / {self.n_runs})"):
                prob = self.init_logistic_regression(C=C)
                prob.train(*train_data)
                self.acc_list.append(prob.predict(*val_data))
            best_idx = np.argmax(self.acc_list)
            best_C = self.search_list[best_idx]
            logging.info(f"Seed {seed} Best C: {best_C}")
            self.c_left, self.c_right = 1e-1 * best_C, 1e1 * best_C
            for step in tqdm(range(self.steps), desc=f"Binary Searching (Seed {seed} | {self.n_runs})"):
                logging.info(f"Binary Searching Round {step}: [c_left {self.c_left} | c_right {self.c_right}]")
                self.binary_search(train_data, val_data, seed, step)
            self.best_c_weights_list.append(self.c_left)
        best_c = np.mean(self.best_c_weights_list)
        classifier = self.init_logistic_regression(C=best_c)
        classifier.train(*train_data)
        val_acc = classifier.predict(*val_data)
        test_acc = classifier.predict(*test_data)
        logging.info(f"Final searched C{best_c} Val accuracy: {val_acc} Test accuracy: {test_acc:.2f}")
        logging.info("-" * 50)


class ScikitLogisticRegression:
    def __init__(self, probe=None) -> None:
        self.probe = probe
        self.train_flag = False

    def train(self, features, labels):
        self.probe.fit(features, labels)
        self.train_flag = True

    def predict(self, features, labels):
        if not self.train_flag:
            raise ValueError("Model must be trained first")
        predictions = self.probe.predict(features)
        self.preds = predictions
        self.labels = labels
        accuracy = np.mean((labels == predictions).astype(float)) * 100.
        return accuracy

    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(self.labels, self.preds)
