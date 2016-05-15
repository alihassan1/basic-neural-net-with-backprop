function out = sigmoid(z)
    out = 1.0 ./(1.0 + exp(-z));
end