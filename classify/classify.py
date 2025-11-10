from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from dlordinal.losses import TriangularLoss,CDWCELoss
from imblearn.over_sampling import SMOTE



### Method to classify a dataset with cross validation and compute relevant metrics using an sklearn model ###
def classify_sklearn(X, y, model, normalize=True, cv_splitter=StratifiedKFold(n_splits=5, shuffle=True), 
                     groups=None, return_preds=False, use_smote=False, smote_k_neighbors=5):
    outputs = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    X_reshaped = np.reshape(X, (X.shape[0], -1))
    
    for train, test in cv_splitter.split(X_reshaped, y, groups=groups):
        train_data = X_reshaped[train]
        test_data = X_reshaped[test]
        y_train = y[train]
        y_test = y[test]

        # Optional normalization
        if normalize:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(train_data)
            train_data = scaler.transform(train_data)
            test_data = scaler.transform(test_data)

        # Optional SMOTE oversampling (only on training data)
        if use_smote:
            sm = SMOTE(k_neighbors=smote_k_neighbors)
            train_data, y_train = sm.fit_resample(train_data, y_train)
        # Train and evaluate
        model.fit(train_data, y_train)
        y_pred = model.predict(test_data)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        outputs.extend([(pred, real, index) for pred, real, index in zip(y_pred, y_test, test)])
    
    # Aggregate and sort predictions
    outputs = sorted(outputs, key=lambda x: x[2])
    outputs = [(pred, real) for pred, real, _ in outputs]

    # Compute mean metrics
    metrics_dict = {
        'mean_accuracy': np.mean(accuracies),
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),
        'mean_f1': np.mean(f1_scores),
    }

    if return_preds:
        metrics_dict['predictions'] = outputs

    return metrics_dict
    

### Method to classify a dataset with cross validation and compute relevant metrics using a pytorch model ###
def classify_torch(X, y, model_class, args=(), kwargs = {}, bayesian = False, cv_splitter = StratifiedKFold(n_splits=5, shuffle=True), groups=None, return_preds = False, batch_size = 32, learning_rate = 0.01, num_epochs = 15, criterion = nn.CrossEntropyLoss()):
    predictions = []
    differences = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    #criterion = TriangularLoss(base_loss=nn.CrossEntropyLoss(), num_classes=kwargs['num_classes']).to(device)
    criterion = CDWCELoss(num_classes=kwargs['num_classes']).to(device)
    data_tensor = torch.tensor(np.expand_dims(X,axis=1),dtype=torch.float32)
    labels_tensor = torch.tensor(y,dtype=torch.long)
    for train, test in cv_splitter.split(data_tensor,labels_tensor,groups=groups):
        try:
            model = model_class(*args,**kwargs)
        except Exception as e:
            print(e)
            print("Invalid model initialization")
            return -1
        model.to(device)
        train_data, test_data = data_tensor[train], data_tensor[test]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_size = train_data.size()
        test_size = test_data.size()
        train_data = train_data.view(train_data.size(0),-1).cpu().numpy()
        test_data = test_data.view(test_data.size(0),-1).cpu().numpy()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        train_data = torch.tensor(train_data,dtype=torch.float32).view(train_size)
        test_data = torch.tensor(test_data,dtype=torch.float32).view(test_size)
        train_labels, test_labels = labels_tensor[train], labels_tensor[test]
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if bayesian:
                    kl = get_kl_loss(model)
                    loss = loss + kl / inputs.size(0)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
        with torch.no_grad():
            curr_idx = 0
            accuracies_temp = []
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, -1)
                preds = preds.cpu().numpy()
                targets = targets.cpu().numpy()
                differences.extend([abs(pred-real) for pred, real in zip(preds,targets)])
                accuracies.append(accuracy_score(targets,preds))
                accuracies_temp.append(accuracy_score(targets,preds))
                precisions.append(precision_score(targets,preds,zero_division=0,average='weighted'))
                recalls.append(recall_score(targets,preds,zero_division=0,average='weighted'))
                f1_scores.append(f1_score(targets,preds,zero_division=0,average='weighted'))
                predictions.extend([(pred,real,index) for pred, real, index in zip(preds,targets,test[curr_idx:curr_idx+len(targets)])])
                curr_idx = len(targets)
            print(f"Mean accuracy for current fold: {np.mean(accuracies_temp)}")
    predictions = sorted(predictions, key=lambda x: x[2])
    predictions = [(pred,real) for pred, real, _ in predictions]
    mean_cv_accuracy = np.mean(accuracies)
    best_fold_accuracy = np.max(accuracies)
    worst_fold_accuracy = np.min(accuracies)
    mean_cv_precision = np.mean(precisions)
    best_fold_precision = np.max(precisions)
    worst_fold_precision = np.min(precisions)
    mean_cv_recall = np.mean(recalls)
    best_fold_recall = np.max(recalls)
    worst_fold_recall = np.min(recalls)
    mean_cv_f1 = np.mean(f1_scores)
    best_fold_f1 = np.max(f1_scores)
    worst_fold_f1 = np.min(f1_scores)
    mean_cv_difference = np.mean(differences)
    median_cv_difference = np.median(differences)
    metrics_dict = {
        'mean_accuracy': mean_cv_accuracy,
        'best_accuracy': best_fold_accuracy,
        'worst_accuracy': worst_fold_accuracy,
        'mean_precision': mean_cv_precision,
        'best_precision': best_fold_precision,
        'worst_precision': worst_fold_precision,
        'mean_recall': mean_cv_recall,
        'best_recall': best_fold_recall,
        'worst_recall': worst_fold_recall,
        'mean_f1': mean_cv_f1,
        'best_f1': best_fold_f1,
        'worst_f1': worst_fold_f1,
        'mean_difference': mean_cv_difference,
        'median_difference': median_cv_difference
    }
    if return_preds:
        metrics_dict['predictions'] = predictions
    return metrics_dict