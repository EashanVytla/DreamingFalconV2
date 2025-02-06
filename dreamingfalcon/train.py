import torch
from dreamingfalcon.world_model import WorldModel
from dreamingfalcon.dataloader import Pipeline
import os
from tqdm import tqdm
from dreamingfalcon.utils import AttrDict
import yaml
from torch.utils.tensorboard import SummaryWriter
from dreamingfalcon.sequence_scheduler import AdaptiveSeqLengthScheduler
import pandas as pd

model_directory = "models/1-31-2-Synthetic"
data_directory = "data/1-31-2-Synthetic"
log_directory = "runs/1-31-2"

def compute_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def compute_weight_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def print_gradient_norms(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.data.norm().item()
            print(f"{name:30s} | grad norm: {grad_norm:.2e} | param norm: {param_norm:.2e}")

def main():
    with open('config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)

    config = AttrDict.from_dict(config_dict)

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")  # Use the GPU
    #     print("Using GPU:", torch.cuda.get_device_name(0)) 
    # else:
    device = torch.device("cpu")  # Use the CPU
    print("Using CPU")

    model = WorldModel(config, device).to(device)

    print(compute_weight_norm(model))

    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.training.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    print(f"Using learning rate: {config.training.lr}")
    if config.training.cos_lr:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=25, eta_min=config.training.min_lr, verbose=True
        )

    writer = SummaryWriter(log_directory)

    seq_scheduler = AdaptiveSeqLengthScheduler(initial_length=config.training.init_seq_len, max_length=config.training.max_seq_len, patience=config.training.seq_patience, threshold=config.training.seq_sch_thresh, model=model, config=config)

    pipeline = Pipeline(os.path.join(data_directory, "train/states.csv"), os.path.join(data_directory, "train/actions.csv"), os.path.join(data_directory, "val/forces.csv"), seq_len=1)
    dataloader = pipeline.read_csv(batch_size=config.training.batch_size)
    
    model.compute_normalization_stats(dataloader)
    # epoch = 0

    for epoch in range(config.training.num_epochs):
    # while seq_scheduler.current_length <= config.training.max_seq_len:
        # print(f"\nEpoch {epoch} parameter norms:")
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.data.norm().item():.6f}")
        print(f"Seq Length: {seq_scheduler.current_length}")
        pipeline = Pipeline(os.path.join(data_directory, "train/states.csv"), os.path.join(data_directory, "train/actions.csv"), os.path.join(data_directory, "val/forces.csv"), seq_len=seq_scheduler.current_length)
        dataloader = pipeline.read_csv(batch_size=config.training.batch_size)


        for batch_count, (states, actions, forces) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            states = states.to(device)
            actions = actions.to(device)
            forces = forces.to(device)
            optimizer.zero_grad()

            # pred_forces, pred_traj = model.rollout(states[:,:,0], actions, seq_scheduler.current_length)

            # loss = model.loss(pred_traj[:,:,1:], states[:,:,1:])
            # print(forces.shape)
            # loss = model.loss(pred_forces[:, :, 0], forces[:, :, 0])
            # loss = model.loss(pred_traj[:,:,-1], states[:,:,-1])

            pred_forces, x_t = model.predict(states[:, :, 0], actions[:, :, 0])

            # print(f"Predicted forces shape: {pred_forces.shape}")
            # print(f"Ground truth forces shape: {forces.shape}")
            loss, loss_vec = model.loss(pred_forces, forces[:, :, 0])

            # print(f"Loss: {loss}")

            loss.backward()
            # grad_norm = compute_gradient_norm(model)
            # if grad_norm == 0:
            #     print(torch.max(states))
            #     print(torch.min(states))
            # else:
            #     print(f"Gradient Norm: {grad_norm}")
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Track total gradient norm
        grad_norm = compute_gradient_norm(model)
        writer.add_scalar("Gradients/total_norm", grad_norm, epoch)
        # Track total weight norm
        weight_norm = compute_weight_norm(model)
        writer.add_scalar("Weights/total_norm", weight_norm, epoch)
        writer.add_scalar("Loss/fx", loss_vec[0], epoch)
        writer.add_scalar("Loss/fy", loss_vec[1], epoch)
        writer.add_scalar("Loss/fz", loss_vec[2], epoch)
        writer.add_scalar("Loss/Mx", loss_vec[3], epoch)
        writer.add_scalar("Loss/My", loss_vec[4], epoch)
        writer.add_scalar("Loss/Mz", loss_vec[5], epoch)
        writer.add_scalar("Loss/train", loss, epoch)

        if config.training.cos_lr:
            lr_scheduler.step()
        
        seq_scheduler.step(loss.item(), optimizer)
        # epoch += 1

    state = {
        "state_dict": model.state_dict(),
        "states_mean": model.states_mean,
        "states_std": model.states_std,
        "actions_mean": model.actions_mean,
        "actions_std": model.actions_std
    }

    torch.save(state, os.path.join(model_directory, "model.pt"))
    writer.close()
    print("Model saved!")

if __name__ == "__main__":
    main()