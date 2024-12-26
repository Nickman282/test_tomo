from common import *

param_dict = load_params(os.path.join(os.getcwd(), "params.json"))

filepaths = param_dict["train_filepaths"]
diameter_bounds = [param_dict["diam_lowq"], param_dict["diam_highq"]]

processor_cl = Processor(filepaths, pmin=0, pmax=1, 
                         diameter_bounds=diameter_bounds)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch = processor_cl.norm_loader(batch_idx=0, batch_size=10)


learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 50


model = VAE().to(device)

batch_tensor = torch.Tensor(batch).to(device)
loss = model(batch_tensor)

print(loss)