import torch
from world_model import WorldModel
from dataloader import Pipeline
import os
from tqdm import tqdm
from utils import AttrDict
import yaml
from torch.utils.tensorboard import SummaryWriter

model_directory = "models/10-14-Synthetic"
data_directory = "data/10-14-Synthetic/train"
log_directory = "runs/10-14"

def main():
    with open('config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)

    config = AttrDict.from_dict(config_dict)

    model = WorldModel(config)

    pipeline = Pipeline(os.path.join(data_directory, "states.csv"), os.path.join(data_directory, "actions.csv"), os.path.join(data_directory, "rewards.csv"))
    dataloader = pipeline.read_csv(batch_size=config.training.batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training.lr)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
        print("Using GPU:", torch.cuda.get_device_name(0)) 
    else:
        device = torch.device("cpu")  # Use the CPU
        print("Using CPU")

    writer = SummaryWriter(log_directory)

    for epoch in range(config.training.num_epochs):
        for batch_count, (states, actions) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):

            for param in model.model.parameters():
                param.requires_grad = True

            pred = model.predict(states, actions)

            loss = model.loss(pred[:-1], states[1:])

            writer.add_scalar("Loss/train", loss, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(model_directory, "model.pt"))
    writer.close()

if __name__ == "__main__":
    main()