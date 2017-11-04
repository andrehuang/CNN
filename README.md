# Interpretable CNN
## Haiwen's approach


### 11.4 updates: uploaded the demo framework: mask_cnn_mnist.py; modified mask_op.py, with forward pass successfully run in the "framework".  __NEXT__: combine forward with corresponding bprop function.
### 10.31 updates: uploaded mask_op, which has the forward pass function changed to be the mask function.

### brief intro to the approach:
My approach is to create a new op with its bprop method and put it into the tensorflow, which will enable auto-training and gradient-computing.

I've already succeeded in creating an relu op with its bprop method. In the ***our_cnn_mnist.py***, I changed the activation function of conv1 to our customized tf_relu, and now it can run and get gradients successfully.

Under this approach, our next step is to modify the ***our_py_func.py***, change the demo function into our customized mask and its bprop method. Hopefully we could get it done soon!
