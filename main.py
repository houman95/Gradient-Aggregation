import multiprocessing as mp
import random
from collections import OrderedDict
from functools import partial
from typing import List
from multiprocessing import Pool
import numpy as np
import torch
from models_and_data import ResNet18, load_cifar10_splits, test_fn, train_fn
from torch.utils.data import DataLoader
from utils import set_nested_attr
import pickle  # write and read results into/from a file
import time
import copy
import os
import logging
import pickle
import torch.nn as nn
import concurrent.futures
# ... rest of your code ...

from collections import defaultdict
from multiprocessing import Manager
temp_log_file = 'flcifar_fl_logs0.log'
logging.basicConfig(filename=temp_log_file, level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')


#########################################################
########### Server and Client Implementation ############
#########################################################

class Client():
    def __init__(
            self,
            client_id: int,
            localaccur: float,
            train_fn: callable,
            trainloader_fn: callable,
            trainset: torch.utils.data.Dataset,
            test_fn: callable,
            local_model: nn.Module,  # Pass a pre-trained model instance for local_model
            **kwargs,
    ):
        self.client_id = client_id
        self.localaccur = localaccur,
        self.train_fn, self.test_fn = train_fn, test_fn
        self.trainset = trainset
        self.trainloader_fn = trainloader_fn
        # Use the provided pre-trained model instance for local_model
        self.local_model = local_model
        self.gradients = None

    def fit(
            self,
            global_model,
            epochs: int,
            rnd: int,
            updated: bool
    ):
        trainloader = self.trainloader_fn(dataset=self.trainset)
        # Copy the parameters of global_model to self.local_model
        #self.local_model.load_state_dict(global_model.state_dict())
        self.local_model = copy.deepcopy(global_model)
        global_model.eval()
        pre_training_state = copy.deepcopy(self.local_model.state_dict())

        cacc = self.test_fn(self.local_model)
        print(f"FL round {rnd} pre | Client {self.client_id} | pre-Training Accuracy: {cacc}")
        self.train_fn(self.local_model, train_loader=trainloader, epochs=epochs)
        del trainloader
        self.localaccur = self.test_fn(self.local_model)
        # Save the client model
        # Save the model parameters as an instance variable
        print(f"FL round {rnd} post | Client {self.client_id} | Training Accuracy: {self.localaccur}")
        #self.gradients = []
        #for param in self.local_model.parameters():
         #   self.gradients.append(param.grad.clone())

        torch.cuda.empty_cache()
        # After training:
        #model_difference = self.compute_model_difference(global_model.state_dict(), self.local_model.state_dict())
        # 3. Compute the model difference
        model_difference = {
            name: self.local_model.state_dict()[name] - global_model.state_dict()[name]
            for name in self.local_model.state_dict().keys()
            if name in global_model.state_dict()
        }
        post_training_model = copy.deepcopy(self.local_model)
        post_training_model.eval()
        #print(f"Sanity check: the accuracy of the post_trained copy is {self.test_fn(post_training_model)} \n")
        # 4. Apply the difference to the pre-training model
        with torch.no_grad():
            reconstructed_model = copy.deepcopy(global_model)
            reconstructed_model.eval()
            #print(f"Sanity check: the accuracy of the pre_trained copy is {self.test_fn(reconstructed_model)}. \n")
            #reconstructed_model.load_state_dict(global_model.state_dict())  # Load pre-training state
            for name, param in reconstructed_model.named_parameters():
                if name in model_difference:
                    param.data.add_(model_difference[name])
                else:
                    print("mismatch \n")
            # Also update the running stats for BatchNorm layers if they exist
            for name, value in reconstructed_model.state_dict().items():
                if name in model_difference and 'num_batches_tracked' not in name:
                    value.copy_(model_difference[name] + value)
                else:
                    print("mismatch found \n")

            # 4. Compare the accuracies
            reconstructed_model.eval()
        post_training_accuracy = self.test_fn(post_training_model)
        reconstructed_accuracy = self.test_fn(reconstructed_model)

        # 5. Print the accuracies to compare them
        print(f"Post-training Model Accuracy: {post_training_accuracy}")
        print(f"Reconstructed Model Accuracy: {reconstructed_accuracy}")
        print(f"local Model Accuracy: {self.localaccur}")
        # 5. Compare resulting model parameters to the post-training model parameters
        for name, param in self.local_model.named_parameters():
            if name in model_difference:
                original_param = reconstructed_model.state_dict()[name]
                relative_difference = torch.abs(param.data - original_param) / torch.abs(original_param).clamp(min=1e-6)
                if not torch.all(relative_difference > 1e-3):
                    print(f"Mismatch found in parameter: {name}, relative difference is {relative_difference} ")

        return model_difference


class Server():
    def __init__(
            self,
            clients: List[Client],
            rounds: int,
            global_model: torch.nn.Module,
            client_transmission_prob: float = 0.01,
            test_fn: callable = None,
            **kwargs
    ):
        self.clients = clients
        self.rounds = rounds
        self.global_model = global_model
        self.client_transmission_prob = client_transmission_prob
        self.test_fn = test_fn
        self.results = []
        self.elapsed_times = []
        self.updated = False
        self.client_results = {}  # Initialize this to store client accuracies for each round
        self.saved_global_model_parameters = {}  # Store global model parameters for each round
        #self.manager = Manager()
        #self.client_results = self.manager.dict()

    def fedavgparams(self, clientlist):
        # Create a deep copy of the first model
        self.global_model.resnet18 = copy.deepcopy(self.clients[clientlist[0]].local_model.resnet18)

        # Initialize the parameters of the average model with zeros
        for param in self.global_model.parameters():
            param.data.fill_(0)

        # Sum up the parameters of all client models
        for clientid in clientlist:
            for avg_param, client_param in zip(self.global_model.parameters(), self.clients[clientid].local_model.resnet18.parameters()):
                avg_param.data += client_param.data

        # Divide by the number of clients to get the average
        num_clients = len(clientlist)
        for avg_param in self.global_model .parameters():
            avg_param.data /= num_clients

    def aggregate_gradients(self, client_gradients_list):
        aggregated_gradients = []
        for gradient_list_tuple in zip(*client_gradients_list):
            aggregated_gradients.append(torch.mean(torch.stack(gradient_list_tuple), dim=0))
        return aggregated_gradients

    def client_execution(self, client, rnd, epochs):
        # Create a new model instance and load the original state
        local_global_model = copy.deepcopy(self.global_model)
        # global_model_state_dict_original = self.saved_global_model_parameters.get(rnd - 1, None)  # Use the previously saved state

        # local_global_model.load_state_dict(global_model_state_dict)
        # Train the client model
        cacc = client.fit(
            global_model=local_global_model,
            epochs=epochs,
            rnd=rnd,
            updated=self.updated
        )
        torch.cuda.empty_cache()
        # Decide if this client's trained model should be transmitted
        # transmit_status = np.random.uniform(0, 1) < self.client_transmission_prob
        # if transmit_status:
        #    return (client.client_id, cacc, True)
        # else:
        self.client_results[client.client_id] = cacc
        return (client.client_id, cacc)  # Now we're always returning accuracy, regardless of transmission

    def client_transmission(self):
        if np.random.uniform(0, 1) < self.client_transmission_prob:
            return True
        return False
    def client_training(self, client_id,epochs,rnd):
        client = self.clients[client_id]
        local_global_model = copy.deepcopy(self.global_model)
        return client.fit(global_model=local_global_model, epochs=epochs, rnd=rnd, updated=self.updated)

    def update_global_model(self, aggregated_difference):
        # Update the global model parameters with the aggregated difference
        with torch.no_grad():  # Disable gradient computation
            for name, param in self.global_model.named_parameters():
                if name in aggregated_difference:
                    # Subtract the difference because in SGD, we move against the gradient
                    param.data.add_(aggregated_difference[name])

    def average_model_differences(self, model_differences):
        aggregated_difference = {}
        for param_name in model_differences[0]:
            aggregated_difference[param_name] = sum([diff[param_name] for diff in model_differences]) / len(model_differences)
        return aggregated_difference

    def fit(self, transmission_attempts=1, num_processes=16):

        print(f'transmission_attemts are {transmission_attempts} \n')
        for rnd in range(1, self.rounds + 1):

            print(f"############## ROUND {rnd} ##############")
            successful_reception_ids_set = set()
            transmission_attempts = int(transmission_attempts)
            for attempt in range(transmission_attempts):
                transmission_results = [self.client_transmission() for _ in self.clients]
                successful_transmission_ids = [cid for cid, decision in enumerate(transmission_results) if decision]
                if len(successful_transmission_ids) == 1:
                    successful_reception_ids_set.update(successful_transmission_ids)
            successful_reception_ids = list(successful_reception_ids_set)

            model_differences = []

            # Sequential loop to collect model differences from each client
            #for client_id in successful_reception_ids[0]:
            client_id = successful_reception_ids[0]
            client = self.clients[client_id]
            local_global_model = copy.deepcopy(self.global_model)
            model_difference = client.fit(
                global_model=local_global_model,
                epochs=1,
                rnd=rnd,
                updated=self.updated
            )
            model_differences.append(model_difference)  # Collect the model differences
            gacc = self.test_fn(self.global_model)
            self.results.append({
                'round': rnd,
                'global_acc': gacc,
            })
            print(f"FL round {rnd} | Server pre-train | Acc: {gacc}")

            # Aggregate the model differences
            averaged_difference = self.average_model_differences(model_differences)

            self.update_global_model(averaged_difference)
            # Test the global model
            gacc = self.test_fn(self.global_model)
            self.results.append({
                'round': rnd,
                'global_acc': gacc,
            })

            logging.info(f"FL round {rnd} | Server post-train | Acc: {gacc}")
            print(f'Round {rnd} finished Server post-train | Acc: {gacc}\n')
            torch.cuda.empty_cache()



########################
### Hyperparameters ####
########################
def run_experiment(NUM_CLIENTS=200, FL_ROUNDS=10, FrameWidth=10, output_file="results.pkl"):
    # mp.set_start_method('spawn')
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    VERBOSE = False
    BATCH_SIZE = 16
    NUM_WORKERS_DATALOADER = 0  # int(mp.cpu_count() * 0.3)
    CLIENT_TRANSMISSION_PROB = 1 / NUM_CLIENTS
    num_processes = 8
    ###################
    ### FL Training ###
    ###################
    trainloader_fn = partial(DataLoader, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_DATALOADER)
    testloader_fn = partial(DataLoader, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_DATALOADER)
    train_splits, test_dataset = load_cifar10_splits(NUM_CLIENTS)
    testloader = testloader_fn(test_dataset)
    global_model = ResNet18().to(DEVICE)
    initial_accur = test_fn(global_model,test_loader = testloader)
    # global_model = global_model.to('cuda').half()
    clients = []
    for cid in range(NUM_CLIENTS):
        curr_client = Client(
            client_id=cid,
            local_model=global_model,
            localaccur= initial_accur,
            trainloader_fn=trainloader_fn,
            trainset=train_splits[cid],
            train_fn=partial(train_fn, verbose=VERBOSE),
            test_fn=partial(test_fn, test_loader=testloader, verbose=VERBOSE),
        )
        clients.append(curr_client)
    server = Server(
        clients=clients,
        rounds=FL_ROUNDS,
        global_model=global_model,
        client_transmission_prob=CLIENT_TRANSMISSION_PROB,
        test_fn=partial(test_fn, test_loader=testloader, verbose=VERBOSE),
    )
    server.fit(transmission_attempts=FrameWidth, num_processes=num_processes)
    print("server fit done \n")
    del global_model
    print("global model deleted \n")
    results_data = {
        "simulation_results": server.results,
    }
    print("end of the program")
    return results_data


if __name__ == '__main__':
    mp.set_start_method('spawn')
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_CLIENTS = 100
    FL_ROUNDS = 50
    # Iterate over frame widths from 10 to 200 with a precision of 10
    for frame_width in range(10, 12, 10):
        FrameWidth = int(frame_width)
        output_file = f"results_clients_{NUM_CLIENTS}_rounds_{FL_ROUNDS}_framewidth_{FrameWidth}.pkl"
        results = run_experiment(NUM_CLIENTS, FL_ROUNDS, FrameWidth)
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)