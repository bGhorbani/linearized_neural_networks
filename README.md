# linearized_neural_networks
This repository contains the code for the paper "When Do Neural Networks Outperform Kernel Methods?" (https://arxiv.org/abs/2006.13409). 

To train NN, RF, NT, or CNNs the main file of interest is train.py. This file includes 
the necessary functionalities for training / measuring the performance of the various 
models in question. The train.py can be run as

python train.py --learning_rate=${rate} --max_ncg_iters=0 --max_cg_iters=750 --exp_name=${ename} --model=${model} --num_units=${units} 
--reg_index=${reg} --max_batch_size=${max_batch_size} --num_layers=${layers} --dataset=${dataset} --noise_ind=${noise_ind}

The flags are documented inside the code. The values used for the experiments are listed in the appendix. max_ncg_iters is the number
of Newton steps to perform during training. It is a legacy functionality in the code and should be set to zero. 

train.py has the following dependencies:
* optimization_utils: Auxilliary tools for training models with CG
* neural_networks: The code for implementation of the models in TensorFlow
* rf_optimizer: The specialized code for training large RF models
* preprocess: The code for data preprocessing and noise addition

Note that to successfully run the code, the original datasets (CIFAR-10, FMNIST, synthetic data) have to be downloaded and placed in the appropriate directory as described in preprocess.py. The directory list in directories.txt has to be updated to reflect the new setting. The synthetic data can be regenerated using the included Iphyton notebook.

===========================================================================

KRR experiments are performed via Kernel_Fit.py and CNT_Kernel_Gen / NTK_Kernel_Gen / RF_Kernel_Gen:

CNT_Kernel_Gen uses JAX and neural_tangents library to generate convolutional neural tangent kernels in a parallelized manner. 
This kernel is then saved to the disk. The generation process is computationally heavy. For a single noise level, this process
can take around 300 GPU hours.

NTK_Kernel_Gen uses JAX and neural_tangents library to generate NT kernel for multi-layer fully-connected networks. The process
is rather computationally light and can be done on CPU.

RF_Kernel_Gen generates kernels for finite-width RF models that are too large to fit directly. We use this functionality to fit
RF models with width 4.2 \times 10^6.

Kernel_Fit contains the code for generating the kernel (simple models) or reading the kernel from the disk (computationally 
challenging models) and fitting KRR. The results are saved to the disk for plotting.

All these files have preproccess.py as a dependency. Note that CNT_Kernel_Gen and NTK_Kernel_Gen requires python3.6 to run. 
