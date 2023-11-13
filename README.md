# Gradient-Aggregation

We have some clients with local dataset. They train a resnet18 model. Then they compute the parameter difference between the trained model and pre-training model, and send this difference to the server.
The server takes the average of the received differences and applies it to the global model. Then, in the next FL round, sends the updated global model to the clients.
