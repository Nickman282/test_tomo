import torch

def model_train(model, optimizer, data_loader, device, num_batches, 
                batch_size=100):

    model.train()
        
    running_loss = 0.0
    for batch_idx in range(num_batches):

        x = data_loader(batch_idx, batch_size)
        x = torch.Tensor(x).to(device=device)

        print(x.shape)

        optimizer.zero_grad()

        loss = model(x)

        print(loss)

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    
        
        optimizer.step()  # Update the model parameters

    return None
