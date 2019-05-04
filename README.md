# LegoNet_private

This code is the implementation of ICML2019 paper [LegoNet: Efficient Convolutional Neural Networks with Lego Filters]()

## Run

```python
python train.py
```

You should achieve an VGG16 with 93.88% accuracy on CIFAR10 dataset, the lego filters occupy ~3.8M parameters.

## LegoConv2d

```python
self.lego = nn.Parameter(nn.init.kaiming_normal_(torch.rand(self.n_lego, self.basic_channels, self.kernel_size, self.kernel_size)))
self.aux_coefficients = nn.Parameter(init.kaiming_normal_(torch.rand(self.n_split, self.out_channels, self.n_lego, 1, 1)))
self.aux_combination = nn.Parameter(init.kaiming_normal_(torch.rand(self.n_split, self.out_channels, self.n_lego, 1, 1)))
```

lego: Lego Filters

aux_coefficients: combination coefficients used during combination

aux_combination: combination index

## Note

The aux_coefficients and aux_combination should be saved as sparse matrix for saving memory. This code does not include this part.
