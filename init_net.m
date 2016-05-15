function net = init_net(layers, MODE)
% Assumes 1st layer is the input layer
% [784, 15, 10] - 3 layers having 784, 15, 10 neurons repectively

num_layers = size(layers,2);
dropout_rate = ones(1, num_layers);
net.use_dropout = 0;
net.num_layers = num_layers-1;

if strcmp(MODE, 'dropout')
    net.use_dropout = 1;
    for i=2:num_layers-1
        dropout_rate(i) = 0.5;
    end    
end

for i=2:num_layers
    for j=1:layers(i)

        net.layers{i-1}.w(j,:) = rand(layers(i-1),1);
        net.layers{i-1}.b(j,1) = rand;
        
%         net.layers{i-1}.w(j,:) = 2 .* rand(layers(i-1),1) - 1;
%         net.layers{i-1}.b(j,1) = 2 * rand - 1;

%         net.layers{i-1}.w(j,:) = (2 .* rand(layers(i-1),1) - 1) ./ sqrt(layers(i-1));
%         net.layers{i-1}.b(j,1) = (2 * rand - 1) ./ sqrt(layers(i-1));

        net.layers{i-1}.dropout_rate = dropout_rate(i);
        net.layers{i-1}.drop = ones(layers(i),1);

    end   
end

end