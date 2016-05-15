function net = dropout(net, i, batch_size)
%  This function deactivates neurons of ith layer in the network 
%  for an entire batch. The output is a matrix ‘drop’ of size 
%  [n_neurons X batch_size], where ‘n_neurons’ is number of neurons in 
%  the ‘i’th layer.  ‘drop’ would contains ‘zeros’ for dropped neurons 
%  and ‘ones’ otherwise.

n_neurons = length(net.layers{i}.b);
net.layers{i}.drop = ones(n_neurons,batch_size);
for j = 1:batch_size    
    idx = randperm(n_neurons, round(n_neurons*net.layers{i}.dropout_rate));
    net.layers{i}.drop(idx, j) = 0;
end

end