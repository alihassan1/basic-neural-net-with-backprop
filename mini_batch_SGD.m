function [net, avg_cost] = update_mini_batch_SGD(net, images, l, learning_rate)
%  Train the neural network using mini-batch stochastic gradient descent.  
%  The 'net' is a struct that defines the configuration of our neural 
%  network; such as, how many layers it has, how many neurons each layer 
%  contains, and their initialized weights and biases. The “images” of course
%  contains training batch of input examples and “lables” contains class or 
%  category label for each training example. 
% 
%  This function returns an updated network (net) and average cost (avg_cost) 
%  computed on batch of examples

% Preparing labels for the entire batch
num_imgs = size(images, 2);
labels = zeros(length(net.layers{end}.b), num_imgs);
for i = 1:num_imgs
    labels(l(i)+1,i) = 1;
end

%% backprop
[dw, db, cost] = backprop(net, images, labels);

%% update
for i = 1:length(net.layers)
    net.layers{i}.w = net.layers{i}.w - (learning_rate/num_imgs) * dw{i};  
    net.layers{i}.b = net.layers{i}.b - (learning_rate/num_imgs) * sum(db{i},2);    
end

avg_cost = sum(cost)/num_imgs;

end
