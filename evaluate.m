function accuracy = evaluate(net, test_images, test_labels, show_confusion_matrix)

if nargin < 4
    show_confusion_matrix = false;
end

scores = feedforward(net, test_images);
[v, p_labels]  = max(scores);
p_labels = p_labels -1;

categories = unique(test_labels);
num_categories = length(categories);

%Confusion Matrix
confusion_matrix = zeros(num_categories, num_categories);
for i=1:length(p_labels)
    r = find(categories == test_labels(i));
    c = find(categories == p_labels(i));
    confusion_matrix(r, c) = confusion_matrix(r, c) + 1;
end

tests_per_cat = length(test_labels) / num_categories;
confusion_matrix = confusion_matrix ./ tests_per_cat;   
accuracy = mean(diag(confusion_matrix));

if show_confusion_matrix == true
    fprintf('Accuracy is %.3f\n', accuracy);
    figure;imagesc(confusion_matrix, [0 1]); 
end

end