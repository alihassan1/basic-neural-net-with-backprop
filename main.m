clc;close all; clear;
%% Load data

train_data_path   = 'C:/Data/MNIST/train-images.idx3-ubyte';
train_labels_path = 'C:/Data/MNIST/train-labels.idx1-ubyte';

test_data_path    = 'C:/Data/MNIST/t10k-images.idx3-ubyte';
test_labels_path  = 'C:/Data/MNIST/t10k-labels.idx1-ubyte';

%  Loading Images
%  helper functions are borrowed from UFLDL tutorial 
%  http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset

train_images = loadMNISTImages(train_data_path);
train_labels = loadMNISTLabels(train_labels_path);

test_images = loadMNISTImages(test_data_path);
test_labels = loadMNISTLabels(test_labels_path);

imW = 28; imH = 28;

%% Preprocessing the data

% zero center data
mean_img = mean(train_images,2);
train_images = gsubtract(train_images,  mean_img);
test_images = gsubtract(test_images,  mean_img);

%% generate train and validation set

total_train_examples = size(train_images, 2);
num_valid_examples = round(total_train_examples*.1);

% randomly select validation set
vidxs = randperm(total_train_examples, num_valid_examples);
valid_imgs = train_images(:,vidxs);
valid_labels = train_labels(vidxs);

tidxs = 1:total_train_examples;
tidxs(vidxs) = 0;
tidxs = tidxs > 0;

train_images = train_images(:,tidxs);
train_labels = train_labels(tidxs);

num_train_examples = size(train_images,2);

%% Stochastic gradient descent

% parameters
learning_rate = 3;
reg_lr = 0.995;
n_shuffle = 27; %54;
batch_size = 2000; %round(num_train_examples / n_shuffle);
training_epochs = 10000;

MODE = 'normal';
% MODE = 'dropout';

net = init_net([784, 60, 10], MODE);
% load('net.mat');

accuracy = evaluate(net, valid_imgs, valid_labels);
disp(['Training epoch:  ', num2str(0), '  Validation:  ', num2str(accuracy) ...
    '  Cost:  ', num2str(-2), '   Lr: ', num2str(learning_rate)]);

max_accuracy = accuracy;

figure; hold on;
for i = 1:training_epochs    
    tr_acc = [];
    cost = [];
    for j = 1:n_shuffle
        idxs = randperm(num_train_examples, batch_size);
        imgs = train_images(:,idxs);
        lbs = train_labels(idxs);
        [net, cost(j)] = mini_batch_SGD(net, imgs, lbs, learning_rate);
        tr_acc(j) = evaluate(net, imgs, lbs, false);
    end
    
    avg_cost = sum(cost)/n_shuffle;
    accuracy = evaluate(net, valid_imgs, valid_labels);
    
    test_error(i) = 1 - accuracy;
    avg_train_error(i) = 1 - sum(tr_acc)/n_shuffle;
    
    plot(1:i, avg_train_error, '-+r');
    plot(1:i, test_error, '-+g');
    legend('Training error', 'Validation error');
    drawnow;
    
    if max_accuracy < accuracy
        save('net.mat', 'net', 'accuracy');
        max_accuracy = accuracy;
        disp('net saved..');
    end
    
    disp(['Training epoch:  ', num2str(i), '  Validation:  ', num2str(accuracy) ...
        '  Cost:  ', num2str(avg_cost), '   Lr: ', num2str(learning_rate)]);
    
%     if learning_rate > 2
%         learning_rate = learning_rate * reg_lr;
%     end
end

%% Evaluate learned network on test data, and show confusion matrix
load('net.mat');
accuracy = evaluate(net, test_images, test_labels, true);