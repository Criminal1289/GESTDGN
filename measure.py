#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import torch
def GetRMSE(y, y_hat):
    y = torch.flatten(y)
    y_hat = torch.flatten(y_hat)
    RMSE = (y - y_hat) ** 2
    RMSE = torch.flatten(RMSE)
    RMSE = sum(RMSE)/len(RMSE)
    RMSE = RMSE ** (1/2)
    #print('fun: GetRMSE')
    RMSE = RMSE.detach().numpy()
    return RMSE

def GetMAP(y, y_hat):
    y = torch.flatten(y)
    y_hat = torch.flatten(y_hat)
    MAP = torch.abs((y - y_hat))
    MAP = sum(MAP)/len(MAP)
    MAP = MAP.detach().numpy()
    return MAP

def GetMAPE(y, y_hat):
    y = torch.flatten(y)
    y_hat = torch.flatten(y_hat)
    MAPE = torch.abs((y - y_hat)/y)
    MAPE = sum(MAPE) / len(MAPE)
    MAPE = MAPE.detach().numpy()
    return MAPE

def R_square(y, y_hat):
    if type(y) is not np.ndarray:
        y = y.detach().numpy()
    if type(y_hat) is not np.ndarray:
        y_hat = y_hat.detach().numpy()
    mean = np.mean(y)
    r = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - mean) ** 2)
    return r

def Skill(y, y_hat):
    if type(y) is not np.ndarray:
        y = y.detach().numpy()
    if type(y_hat) is not np.ndarray:
        y_hat = y_hat.detach().numpy()
    count = []
    for i in range(y.shape[0]):
        account = 0
        for j in range(y.shape[1]):
            account += np.sum(y[i, j, :] * y_hat[i, j, :]) / (
                    np.sum(np.sqrt(y[i, j, :] ** 2)) * np.sum(np.sqrt(y_hat[i, j, :] ** 2)))
        count.append(account / y.shape[1])
    similarity = sum(count) / y.shape[0]
    return similarity

def Norm(DataSet, dim=2, Range=[0, 1]):

    range_ones = Range[1] - Range[0]
    DataSet_min = np.min(DataSet, axis=0)
    DataSet_max = np.max(DataSet, axis=0)
    #print(DataSet_min)
    #print(DataSet_max)
    if dim == 2:
        for i in range(DataSet.shape[1]):
            DataSet[:, i] = (DataSet[:, i] - DataSet_min[i]) / (DataSet_max[i] - DataSet_min[i]) * range_ones + Range[0]
    else:
        DataSet = (DataSet - DataSet_min) / (DataSet_max - DataSet_min) * range_ones + Range[0]
    #DataSet = (DataSet - DataSet_min) / (DataSet_max - DataSet_min)
    #DataSet = torch.tensor(DataSet)
    return DataSet, DataSet_min, DataSet_max
# ================== Rainfall categorical metrics ==================

def _confusion_counts(y, y_hat, thr=0.1):
    """
    Calculate H, M, F, C for rainfall detection.
    y, y_hat: torch.Tensor or np.ndarray, 任意形状
    thr: rainfall threshold (same unit as y)
    """
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(y_hat, np.ndarray):
        y_hat = y_hat.detach().cpu().numpy()

    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)

    obs = y >= thr
    pre = y_hat >= thr

    H = np.sum(obs & pre)        # Hit
    M = np.sum(obs & (~pre))     # Miss
    F = np.sum((~obs) & pre)     # False Alarm
    C = np.sum((~obs) & (~pre))  # Correct Negative

    return H, M, F, C


def GetPOD(y, y_hat, thr=0.1, eps=1e-8):
    """Probability of Detection"""
    H, M, _, _ = _confusion_counts(y, y_hat, thr)
    return H / (H + M + eps)


def GetFAR(y, y_hat, thr=0.1, eps=1e-8):
    """False Alarm Ratio"""
    H, _, F, _ = _confusion_counts(y, y_hat, thr)
    return F / (H + F + eps)


def GetCSI(y, y_hat, thr=0.1, eps=1e-8):
    """Critical Success Index"""
    H, M, F, _ = _confusion_counts(y, y_hat, thr)
    return H / (H + M + F + eps)
