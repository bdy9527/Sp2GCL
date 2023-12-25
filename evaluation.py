import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from ogb.graphproppred import Evaluator

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(hid_dim, out_dim)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        ret = self.linear(x)
        return ret


def node_evaluation(emb, y, train_idx, valid_idx, test_idx, lr=1e-2, weight_decay=1e-4):
    device = emb.device

    nclass = y.max().item() + 1
    logreg = LogReg(emb.shape[1], nclass).to(device)
    train_idx, valid_idx, test_idx, y = train_idx.to(device), valid_idx.to(device), test_idx.to(device), y.to(device)
    opt = torch.optim.Adam(logreg.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0
    pred = None

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()

        logits = logreg(emb)

        preds = torch.argmax(logits[train_idx], dim=1)
        train_acc = torch.sum(preds == y[train_idx]).float() / train_idx.size(0)

        loss = loss_fn(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()


        logreg.eval()
        with torch.no_grad():

            if valid_idx.size(0) != 0:
                val_logits = logreg(emb[valid_idx])
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = torch.sum(val_preds == y[valid_idx]).float() / valid_idx.size(0)
            else:
                train_logits = logreg(emb[train_idx])
                train_preds = torch.argmax(train_logits, dim=1)
                train_acc = torch.sum(train_preds == y[train_idx]).float() / train_idx.size(0)
                val_acc = train_acc
            
            test_logits = logreg(emb[test_idx])
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = torch.sum(test_preds == y[test_idx]).float() / test_idx.size(0)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                if test_acc > eval_acc:
                    eval_acc = test_acc
                    pred = test_preds

            #print('Epoch:{}, train_acc:{:.2f}, val_acc:{:2f}, test_acc:{:2f}'.format(epoch, train_acc, val_acc, test_acc))

    return eval_acc, pred


def ogbg_regression(train_emb, valid_emb, test_emb, train_y):

    base_classifier = Ridge(fit_intercept=True, copy_X=True, max_iter=10000)
    params_dict = {'alpha': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}

    classifier = make_pipeline(Normalizer(),
                GridSearchCV(base_classifier, params_dict, cv=5, scoring='neg_root_mean_squared_error', n_jobs=16, verbose=0))
    classifier.fit(train_emb, np.squeeze(train_y))

    train_pred = classifier.predict(train_emb)
    valid_pred = classifier.predict(valid_emb)
    test_pred  = classifier.predict(test_emb)

    return np.expand_dims(train_pred, axis=1), np.expand_dims(valid_pred, axis=1), np.expand_dims(test_pred, axis=1)


def ogbg_binary_classification(train_emb, valid_emb, test_emb, train_y):

    base_classifier = LogisticRegression(dual=False, fit_intercept=True, max_iter=10000)
    params_dict = {'C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}

    classifier = make_pipeline(StandardScaler(), 
                GridSearchCV(base_classifier, params_dict, cv=5, scoring='roc_auc', n_jobs=16, verbose=0))
    classifier.fit(train_emb, np.squeeze(train_y))

    train_pred = classifier.predict_proba(train_emb)[:, 1]
    valid_pred = classifier.predict_proba(valid_emb)[:, 1]
    test_pred  = classifier.predict_proba(test_emb)[:, 1]

    return np.expand_dims(train_pred, axis=1), np.expand_dims(valid_pred, axis=1), np.expand_dims(test_pred, axis=1)


def ogbg_multi_binary_classification(train_emb, valid_emb, test_emb, train_y):
    
    base_classifier = LogisticRegression(dual=False, fit_intercept=True, max_iter=10000)
    params_dict = {'multioutputclassifier__estimator__C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}

    if np.isnan(train_y).any():
        train_y = np.nan_to_num(train_y)

    #classifier = make_pipeline(StandardScaler(), MultiOutputClassifier(base_classifier, n_jobs=-1))
    pipe = make_pipeline(StandardScaler(), MultiOutputClassifier(base_classifier))
    classifier = GridSearchCV(pipe, params_dict, cv=5, scoring='roc_auc', n_jobs=16, verbose=0)
    classifier.fit(train_emb, train_y)

    train_pred = np.transpose([y_pred[:, 1] for y_pred in classifier.predict_proba(train_emb)])
    valid_pred = np.transpose([y_pred[:, 1] for y_pred in classifier.predict_proba(valid_emb)])
    test_pred  = np.transpose([y_pred[:, 1] for y_pred in classifier.predict_proba(test_emb)])

    return train_pred, valid_pred, test_pred


def ogbg_evaluation(emb, y, split_idx, evaluator, task_type, num_tasks):

    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    train_emb, valid_emb, test_emb = emb[train_idx], emb[valid_idx], emb[test_idx]
    train_y,   valid_y,   test_y   = y[train_idx],   y[valid_idx],   y[test_idx]

    if 'regression' in task_type:

        metric = 'rmse'
        train_pred, valid_pred, test_pred = ogbg_regression(train_emb, valid_emb, test_emb, train_y)

    elif 'classification' in task_type:

        metric = 'rocauc'
        if num_tasks == 1:
            train_pred, valid_pred, test_pred = ogbg_binary_classification(train_emb, valid_emb, test_emb, train_y)
        else:
            train_pred, valid_pred, test_pred = ogbg_multi_binary_classification(train_emb, valid_emb, test_emb, train_y)

    else:
        raise NotImplementedError

    train_score = evaluator.eval({"y_true": train_y, "y_pred": train_pred})[metric]
    valid_score = evaluator.eval({"y_true": valid_y, "y_pred": valid_pred})[metric]
    test_score  = evaluator.eval({"y_true": test_y,  "y_pred": test_pred})[metric]

    return train_score, valid_score, test_score

