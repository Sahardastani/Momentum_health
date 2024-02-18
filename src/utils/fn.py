import torch

nr = 0

def train_fn(train_loader, model, criterion, optimizer):
    loss_epoch = 0
    
    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda:0').float().repeat(1, 3, 1, 1)
        x_j = x_j.to('cuda:0').float().repeat(1, 3, 1, 1)
        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()
        
        if nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")

        loss_epoch += loss.item()
    return loss_epoch

def valid_fn(valid_loader, model, criterion):
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(valid_loader):
        
        x_i = x_i.to('cuda:0').float().repeat(1, 3, 1, 1)
        x_j = x_j.to('cuda:0').float().repeat(1, 3, 1, 1)

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        
        if nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(valid_loader)}]\t Loss: {round(loss.item(),5)}")

        loss_epoch += loss.item()
    return loss_epoch