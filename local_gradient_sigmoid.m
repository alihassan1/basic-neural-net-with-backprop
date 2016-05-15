function a = local_gradient_sigmoid(a)
    a = a .* (1-a);
end