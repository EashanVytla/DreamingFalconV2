import torch
from world_model import WorldModel
from dataloader import Pipeline
import os
from tqdm import tqdm
from utils import AttrDict
import yaml
import csv

model_directory = "models/10-14-Synthetic"
data_directory = "data/10-14-Synthetic/train"
log_directory = "runs/10-14"

def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def main():
    with open('config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)

    config = AttrDict.from_dict(config_dict)

    model = WorldModel(config)

    model_path = os.path.join("models", "10-14-Synthetic", "model.pt")
    model.load_state_dict(torch.load(model_path))

    pipeline = Pipeline(os.path.join(data_directory, "states.csv"), os.path.join(data_directory, "actions.csv"), os.path.join(data_directory, "rewards.csv"))
    dataloader = pipeline.read_csv(batch_size=config.training.batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
        print("Using GPU:", torch.cuda.get_device_name(0)) 
    else:
        device = torch.device("cpu")  # Use the CPU
        print("Using CPU")

    output_file = os.path.join("models", "10-14-Synthetic", "test.csv")

    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        for batch_count, (states, actions) in enumerate(dataloader):
            with torch.no_grad():
                for param in model.model.parameters():
                    param.requires_grad = True

                pred = model.predict(states, actions)

                pred_np = tensor_to_numpy(pred)
                for row in pred_np:
                    csv_writer.writerow(row)



if __name__ == "__main__":
    main()