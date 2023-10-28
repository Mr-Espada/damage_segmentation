import torch
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from dataset.builder import BuildingsDataset
from dataset.preprocessing import preprocessing_fn, get_preprocessing, to_tensor
from models.model import model, loss, metrics, optimizer, DEVICE, EPOCHS, TRAINING, CLASSES, BATCH_SIZE


train_dataset = BuildingsDataset(
    ['tier1', "tier3"],
    preprocessing=get_preprocessing(preprocessing_fn),
    nclasses=len(CLASSES)+1,
)

valid_dataset = BuildingsDataset(
    ['hold'],
    preprocessing=get_preprocessing(preprocessing_fn),
    nclasses=len(CLASSES)+1,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)



train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

train_logs_list, valid_logs_list = [], []

if TRAINING:

    best_iou_score = 0.0 # np.load("best_iou_score.npy")[0]

    for i in range(0, EPOCHS):

        # Perform training & validation
        print('\nEpoch: {}'.format(i+1))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        np.save("train_logs.npy", np.array(train_logs_list))
        np.save("valid_logs.npy", np.array(valid_logs_list))

        # Save model if a better val IoU score i0.943s obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
            np.save("best_iou_score.npy", np.array([best_iou_score]))
    torch.save(model, "./last_model.pth")

