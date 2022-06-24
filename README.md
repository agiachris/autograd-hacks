# autograd-hacks
A simple module attaching hooks to PyTorch models to extract useful autograd quantities. 

**Note:** the functionality of this repository is now implemented in [functorch](https://pytorch.org/functorch/stable) and supports a wider range of network operations. 


## Setup

### System Requirements
Tested on Ubuntu 16.04, 18.04 and macOS Monterey with Python 3.6.

### Installation
This package can be installed via pip as shown below. Its only dependency is PyTorch.
```bash
git clone https://github.com/agiachris/autograd-hacks.git
cd autograd-hacks && pip install .
```


## Usage

### Per-Sample Gradients
PyTorch does not currently support efficient computation of per-sample-gradients. With autograd-hacks, you can obtain them with a single `.backward()` over a scalar.

```python
# Add forward and backward hooks to model
autograd_hacks.add_hooks(model)
output = model(data)
loss_fn(output, targets).backward()
autograd_hacks.compute_grad1()

# per-sample gradients stored in param.grad1
for param in model.parameters():
    assert(torch.allclose(param.grad1.mean(dim=0), param.grad))
```


### Hessians
Computes the Hessian assuming ReLU activations, otherwise producing a Gauss-Newton matrix.

```python
autograd_hacks.backprop_hess(model(data), hess_type='CrossEntropy')
autograd_hacks.compute_hess(model)

# hessian stored in param.hess
for param in model.parameters():
    print(param.hess)
```
