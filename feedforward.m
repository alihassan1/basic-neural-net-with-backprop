function [activations, net] = feedforward(net, x, train_mode)
%  This function as the name suggests feedforwards a batch of examples 
%  through the network. It runs in two modes, ‘train’ and ‘test’. In the 
%  ‘train’ mode, it keeps track of the intermediate results while 
%  feedforwarding; such as inputs and local gradients at each layer, which
%  are required for backpropagation.
%
%  Input:
%  ‘net'            Network struct
%  ‘x’              Batch of images
%  ‘train_mode’     ‘1’ for train mode, ‘0’ otherwise
%
%  Output:
%  ‘activations’    Output of the network at last layer
%  ‘net’            Useful in the ‘train_mode’, contains intermediate results

if nargin < 3
    train_mode = 0;
end

batch_size = size(x,2);
zs = x;
activations = [];

for i = 1:net.num_layers
    net.layers{i}.x = zs;
    zs = gadd(net.layers{i}.w * zs,  net.layers{i}.b);
    activations = sigmoid(zs);
    
    if train_mode == 1
        % droput
        if net.use_dropout == 1 && (i ~= net.num_layers)
            net = dropout(net, i, batch_size);
            activations = activations .* net.layers{i}.drop;
%             activations = activations ./ net.layers{i}.dropout_rate; % Inverted dropout
        end
        
        net.layers{i}.local_gradient = local_gradient_sigmoid(activations);       
    else
        if net.use_dropout == 1
            activations = activations .* net.layers{i}.dropout_rate;
        end
    end
end

end