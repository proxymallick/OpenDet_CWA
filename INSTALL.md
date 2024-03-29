## Installation

Our [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
has step-by-step instructions that install detectron2.
The [Dockerfile](https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile)
also installs detectron2 with a few simple commands.

### Requirements
- Linux or macOS
- Python >= 3.6
- PyTorch 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
	`pip install torch==2.0.1+cu117 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html`
	
	
- Install Detectron2 - `pip install 'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install fvcore==0.1.1.dev200512`



### Common Installation and Run Issues of OpenDet 2

+ Issues with numpy str : change np.str() to str() in the file
	* "opendet_cwa/lib/python3.8/site-packages/detectron2/data/datasets/pascal_voc.py"



+ Issues with "site-packages/geomloss/sinkhorn_divergence.py"
 
   * Change the view function to reshape function and there would be no problems running geomloss
   * View size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.



+  File "/opt/conda/lib/python3.8/site-packages/torch/optim/adamw.py", line 496, in _multi_tensor_adamw
    torch._foreach_mul_(device_params, 1 - lr * weight_decay)
    TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'

* some of the torch versions does not handle None as weight decay so you can add a line to handle:
  if weight_decay not in [None,[]]:
    ....



* File "/opt/conda/lib/python3.8/site-packages/torch/optim/sgd.py", line 291, in _multi_tensor_sgd
  Similar solution as above


+ Undefined torch/aten symbols, or segmentation fault immediately when running the library.
  This may be caused by the following reasons:

	* detectron2 or torchvision is not compiled with the version of PyTorch you're running.

		If you use a pre-built torchvision, uninstall torchvision & pytorch, and reinstall them
		following [pytorch.org](http://pytorch.org).
		If you manually build detectron2 or torchvision, remove the files you built (`build/`, `**/*.so`)
		and rebuild them.

	* detectron2 or torchvision is not compiled using gcc >= 4.9.

	  You'll see a warning message during compilation in this case. Please remove the files you build,
		and rebuild them.
		Technically, you need the identical compiler that's used to build pytorch to guarantee
		compatibility. But in practice, gcc >= 4.9 should work OK.

+ Undefined C++ symbols in `detectron2/_C*.so`:

  * This can happen with old anaconda. Try `conda update libgcc`.

+ Undefined cuda symbols. The version of NVCC you use to build detectron2 or torchvision does
	not match the version of cuda you are running with.
	This often happens when using anaconda's cuda runtime.

+ "Not compiled with GPU support": make sure
	```
	python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
	```
	print valid outputs at the time you build detectron2.

+ "invalid device function" or "no kernel image is available for execution": two possibilities:
  * You build detectron2 with one version of CUDA but run it with a different version.
  * Detectron2 is not built with the correct compute compability for the GPU model.
    The compute compability defaults to match the GPU found on the machine during building,
    and can be controlled by `TORCH_CUDA_ARCH_LIST` environment variable during installation.


	/opt/conda/lib/python3.8/site-packages/detectron2/data/datasets/pascal_voc.py

+ RuntimeError: Couldn't load custom C++ ops. This can happen if your PyTorch and torchvision versions are incompatible, or if you had errors while compiling torchvision from source. F

   * Uninstall torchvision existing version and reinstall torchvision and the torch will consider the compatible version to be installed 


