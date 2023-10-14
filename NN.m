close all, clear all, clc;

% Create and view custom neural networks


% Define one sample: inputs and outputs
% Define and custom network
% Define topology and transfer function
% Configure network
% Train net and calculate neuron output

% Define one sample: inputs and outputs
inputs = [1:6]'; % input vector (6-dimensional pattern)
outputs = [1 2]'; % corresponding target output vector
% Define and custom network
net = network( ...
1, ... % numInputs, number of inputs,
2, ... % numLayers, number of layers
[1; 0], ... % biasConnect, numLayers-by-1 Boolean vector,
[1; 0], ... % inputConnect, numLayers-by-numInputs Boolean matrix,
[0 0; 1 0], ... % layerConnect, numLayers-by-numLayers Boolean matrix
[0 1] ... % outputConnect, 1-by-numLayers Boolean vector
);
% View network structure
view(net);
% Define topology and transfer function
net.layers{1}.size = 5; % number of hidden layer neurons
net.layers{1}.transferFcn = 'logsig'; % hidden layer transfer function
view(net);
% Configure network
net = configure(net,inputs,outputs);
view(net);
% Train net and calculate neuron output
initial_output = net(inputs) % initial network response without 
training
% network training
net.trainFcn ='trainlm';
net.performFcn ='mse'; % Mean Square Error
net = train(net,inputs,outputs);
% network response after training
final_output = net(inputs)
