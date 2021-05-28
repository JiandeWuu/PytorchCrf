import os
import time
import random

import numpy as np

import torch
import torch.utils.data as Data

from sklearn.metrics import confusion_matrix

class Trainer:
    def __init__(self, model, optimizer=None, loss_function=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
    
    def build_data(self, x, y, mask=None, batch_size=1, num_workers=1, seed=None):
        if mask is None:
            mask = torch.ones_like(y)

        train_dataset = Data.TensorDataset(x, mask, y)
        
        loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(not seed is None),
        )
        return loader

    def model_to_loss(self, x, y, mask):
        loss = self.model(x, mask, y)
        return loss

    def train(self, x, y, mask=None, epochs=2, batch_size=1, num_workers=1, print_size=1, seed=None):
        if not seed is None:
            self._seed_torch(seed)

        loader = self.build_data(x, y, mask=mask, batch_size=batch_size, num_workers=num_workers)
        
        start_time = time.time()
        step_size = len(loader)
        last_loss = 1
        loss_array = []
        for epoch in range(epochs):
            stop_count = 0
            epoch_time = time.time()
            for step, (batch_x, batch_mask, batch_y) in enumerate(loader):
                step_time = time.time()

                batch_x = batch_x.cuda()
                batch_mask = batch_mask.cuda()
                batch_y = batch_y.cuda()

                self.optimizer.zero_grad()

                loss = self.model_to_loss(batch_x, batch_y, batch_mask)
                
                loss.backward()
                self.optimizer.step()
                loss_array.append(float(loss))
                if step % print_size == 0 or step == step_size:
                    print('Epoch: %i | Step: %i/%i | Loss: %.2f | time: %.2f s' % (epoch + 1, step + 1, step_size, loss, time.time() - step_time))
                if abs((float(loss) - last_loss) / last_loss) < 0.0001:
                    stop_count += 1
                last_loss = float(loss)
                if stop_count > (min(step_size / 2, 100)):
                    break
            print("Epoch time:", time.time() - epoch_time)

        print('All time:', time.time() - start_time,'s')
        return float(loss), loss_array

    def _seed_torch(self, seed=1029):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def evaluation(self, x, y, mask=None, batch_size=1, num_workers=1, print_size=1, accuracy=False, confusion_matrix_size=False):
        loader = self.build_data(x, y, mask=mask, batch_size=batch_size, num_workers=num_workers)

        # eval
        step_size = len(loader)
        epoch_time = time.time()
        loss_array = []
        true_array = []
        total_array = []
        cnf_matrix = None
        for step, (batch_x, batch_mask, batch_y) in enumerate(loader):
            step_time = time.time()

            batch_x = batch_x.cuda()
            batch_mask = batch_mask.cuda()
            batch_y = batch_y.cuda()

            loss = self.model_to_loss(batch_x, batch_y, batch_mask)

            loss_array.append(float(loss))

            if accuracy or confusion_matrix_size:
                pre_tag = self.model.predict(batch_x).cpu()
                # cnf_matrix
                if confusion_matrix_size:
                    all_pre = np.array(pre_tag).flatten()
                    np_y = batch_y.cpu().numpy().flatten()
                    if cnf_matrix is None:
                        cnf_matrix = confusion_matrix(np_y, all_pre, labels=np.arange(confusion_matrix_size))
                    else:
                        cnf_matrix += confusion_matrix(np_y, all_pre, labels=np.arange(confusion_matrix_size))
                # accuracy
                if accuracy:
                    pre_tag = torch.tensor(pre_tag).cuda()

                    batch_true_num = int(torch.sum(torch.where(torch.tensor(pre_tag) == batch_y, batch_mask, torch.zeros_like(batch_y))))
                    batch_num = int(torch.sum(batch_mask))

                    true_array.append(batch_true_num)
                    total_array.append(batch_num)
                    if step % print_size == 0 or step == step_size:
                        print('Epoch: %i/%i | Loss: %.2f | Accuracy: %.2f | time: %.2f s' % (step + 1, step_size, loss, batch_true_num / batch_num, time.time() - step_time))
            elif step % print_size == 0 or step == step_size:
                print('Epoch: %i/%i | Loss: %.2f | time: %.2f s' % (step + 1, step_size, loss, time.time() - step_time))
        
        output = []
        output.append(sum(loss_array) / len(loss_array))
        if accuracy:
            print("Average Accuracy:", sum(true_array) / sum(total_array))
            output.append(sum(true_array) / sum(total_array))
        if confusion_matrix_size:
            print("cnf_matrix")
            output.append(cnf_matrix)
        print("Average Loss:", output[0])
        print("epoch time:", time.time() - epoch_time)
        return output

    def save_model(self, save_filename = "model.pt"):
        torch.save(self.model, save_filename)
    
    def get_model(self):
        return self.model
