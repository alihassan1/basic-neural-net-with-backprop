function [dw, db, cost ]= backprop(net, x, labels)
%  This function backpropagates a batch of training examples through 
%  the network.
%
%  ‘net'      Network struct
%  ‘x’        Batch of images
%  'labels'  Holds category labels for each training example
%
%  'dw'      Contains derivatives w.r.t ‘w’ at each layer
%  'db'      Contain derivatives w.r.t ‘b’ at each layer
%  'cost'    Total cost for entire batch

%% feedforward
[activations, net] = feedforward(net, x, 1);

%% backword pass
dcost = activations - labels; % derivative on dC/dz
delta = dcost .* net.layers{end}.local_gradient;
n = length(net.layers);

% Derivatives propagating through the top layer
db{n} = delta;
dw{n} = delta * net.layers{end}.x';

% Derivatives propagating through from the 2nd last layer to the 1st layer
% of the network. Therefore, the reverse loop.
for i = length(net.layers)-1:-1:1
    delta = (dw{i+1}' * delta) .* net.layers{i}.local_gradient;
    
    if net.use_dropout == 1
       delta = delta .* net.layers{i}.drop ; 
    end
    
    db{i} = delta;
    dw{i} = delta * net.layers{i}.x';
end

cost = sum(dcost.^2)./2;
end