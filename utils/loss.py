import torch

def get_metrics(keep, target):
    with torch.no_grad():
        TP = (keep * target).nonzero().shape[0]
        FN = (~keep * target).nonzero().shape[0]
        FP = (keep * ~target).nonzero().shape[0]
        # TN = (~keep * ~target).nonzero().shape[0]

        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        IoU = TP / (TP + FP + FN + 1e-7)

    return [round(precision, 4), round(recall, 4), round(IoU, 4)]