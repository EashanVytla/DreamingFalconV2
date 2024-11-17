import torch
from world_model import WorldModel
from dataloader import Pipeline
import os
from tqdm import tqdm
from utils import AttrDict
import yaml
from torch.utils.tensorboard import SummaryWriter

model_directory = "models/11-17-Synthetic"
data_directory = "data/11-17-Synthetic/train"
log_directory = "runs/11-17"

def compute_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
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

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
        print("Using GPU:", torch.cuda.get_device_name(0)) 
    else:
        device = torch.device("cpu")  # Use the CPU
        print("Using CPU")

    model = WorldModel(config, device).to(device)

    pipeline = Pipeline(os.path.join(data_directory, "states.csv"), os.path.join(data_directory, "actions.csv"), seq_len=config.training.seq_len)
    dataloader = pipeline.read_csv(batch_size=config.training.batch_size)
    #optimizer = torch.optim.SGD(model.parameters(), lr=config.training.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='min', factor=0.8, patience=5, verbose=True
    )

    writer = SummaryWriter(log_directory)

    # model.compute_normalization_stats(dataloader)

    for epoch in range(config.training.num_epochs):
        # print(f"\nEpoch {epoch} parameter norms:")
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.data.norm().item():.6f}")

        for batch_count, (states, actions) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            states = states.to(device)
            actions = actions.to(device)
            optimizer.zero_grad()

            pred_traj = model.rollout(states[:,:,0], actions, config.training.seq_len)

            loss = model.loss(pred_traj, states)

            # print(f"Loss: {loss}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Track total gradient norm
        grad_norm = compute_gradient_norm(model)
        writer.add_scalar("Gradients/total_norm", grad_norm, epoch)
    
        writer.add_scalar("Loss/train", loss, epoch)
        scheduler.step(loss.item())

    torch.save(model.state_dict(), os.path.join(model_directory, "model.pt"))
    writer.close()

if __name__ == "__main__":
    main()