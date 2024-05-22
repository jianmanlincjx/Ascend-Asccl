import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader.spatial_coherent_loader import SpatialDataloader
from model.spatial_coherent_model import Spatial_Coherent_Correlation_Learning
import sys
import os
sys.path.append(os.getcwd())
import random
import numpy as np
import torch_npu
from torch_npu.contrib import transfer_to_npu


def fixed_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =  True



if __name__ == "__main__":
    fixed_seed()
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    train_data = SpatialDataloader("train")
    test_data = SpatialDataloader("test")
    model = Spatial_Coherent_Correlation_Learning().npu()
    resnet50_pretrain = torch.load("./pretrain/backbone.pth", map_location="npu")
    model.resnet50.load_state_dict(resnet50_pretrain, strict=False)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=64)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=8)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data_len_train = len(train_dataloader)
    data_len_test = len(test_dataloader)

    iter = 0
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        pos_all = 0.0
        neg_all = 0.0
        iter_epoch = 0

        for batch in train_dataloader:
            neutral_img = batch['source_img'].cuda()
            emotion_img = batch['target_img'].cuda()

            optimizer.zero_grad()
            loss, pos, neg = model(neutral_img, emotion_img)
            loss.backward()
            optimizer.step()
            loss_num = loss.item()
            train_loss += loss_num
            pos_all += pos
            neg_all += neg
            iter += 1
            iter_epoch += 1
            print(f"epoch: {epoch}  iter: {iter}  train_loss: {loss_num:.3f} pos_score: {pos:.3f} neg_score: {neg:.3f}")
            if iter % 100 == 0:
                writer.add_scalar(f"train_iter/Patch_NCELoss_iter", train_loss/iter_epoch, iter)   
                writer.add_scalar(f"train_iter/pos_iter", pos_all/iter_epoch, iter)   
                writer.add_scalar(f"train_iter/neg_iter", neg_all/iter_epoch, iter)   
        writer.add_scalar(f"train_epoch/Patch_NCELoss_epoch", train_loss/data_len_train, epoch)
        writer.add_scalar(f"train_epoch/pos_epoch", pos_all/data_len_train, epoch)
        writer.add_scalar(f"train_epoch/neg_epoch", neg_all/data_len_train, epoch)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'/home/qinjinghui/code/asccl/model_ckpt/{epoch}-128_landmarks_align.pth')
            
        model.eval()
        test_loss = 0.0
        test_pos = 0.0
        test_neg = 0.0
        with torch.no_grad():  
            for batch in test_dataloader:
                neutral_img = batch['source_img'].cuda()
                emotion_img = batch['target_img'].cuda()

                loss, pos, neg = model(neutral_img, emotion_img)
                loss_num = loss.item()
                test_loss += loss_num
                test_pos += pos
                test_neg += neg
                print(f"epoch: {epoch}  iter: {iter}  train_loss: {loss_num} pos_score: {pos} neg_score: {neg}")
            writer.add_scalar(f"test/NCELoss", test_loss/data_len_test, epoch)
            writer.add_scalar(f"test/pos_epoch", test_pos/data_len_test, epoch)
            writer.add_scalar(f"test/neg_epoch", test_neg/data_len_test, epoch)

