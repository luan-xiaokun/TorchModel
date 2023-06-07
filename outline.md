# Problem We Aim to Solve
Fine-tuning a pre-trained model on specific datasets is a very common practice in DL community nowadays.
Developers need to implement the fine-tuning part by hand, define what kind of techniques is used to reuse the pre-trained models, such as the dataset, the optimizer, optimization parameters, training setting, augmentation setting, and different reuse techniques (fine-tuning, distillation, etc).
To provide better reusability and modularity, we propose a DSL for defining and implementing the fine-tuning (reusing) process (simi-)automatically.

# General Idea
This DSL should have the following features:
1. Strong support for PyTorch
2. Support both popular pre-defined model architectures (such as ResNet, VGG, etc) and also custom models (nn.Module)
3. Support both popular pre-defined datasets (such as CIFAR10) and also custom datasets (Stanford Dogs)
4. Support various reuse techniques, such as fine-tuning with different layers frozen, knowledge distillation, re-initialization, pruning, etc.
5. Support for various PyTorch optimizers, and also support for user-defined optimizers (and also optimization parameters)
6. Support for various learning rate schedulers
7. Support for various training loss (including label smoothing)
8. Support for data preprocessing (augmentation)
9. Automatically generate Python code for training
10. Support for early-stopping
11. Pretty training process output (tqdm)
12. Input model, dataset, checkpoint, output path arguments
13. User friendly grammar
14. Also support training from scratch (i.e., no pre-trained model)
15. Allow users to override some parts of the whole procedure
16. [TBD] Automatic Mixed Precision (AMP)
17. [TBD] distributed training
18. [TBD] TensorBoard

# DSL Design
A dsl file describes a training recipe, it should have the following part:
1. Source model/teacher model, i.e., the pre-trained model. When it is empty/un-provided, it means training from scratch
2. Dataset, i.e., which dataset is used to train the model
3. 