loss use "cross-entropy" where
    label_smoothing = 0.1
end

optimizer use "SGD" where
    lr = 0.01
    weight_decay = 2e-5
    momentum = 0.9
end

lr_scheduler use "Step" where
    step_size = 10
    gamma = 0.2
end

dataset use "MNIST" where
    root = "examples/mnist"
    val_ratio = 0.1
    transform = "examples/mynet.py/transform"
end

model use "examples/mynet.py/Net" where
    num_classes = 10
end

training use "plain" where
    checkpoint_path = "examples/model/demo.pt"
    epoch_num = 5
    batch_size = 192
    print_freq = 50
end