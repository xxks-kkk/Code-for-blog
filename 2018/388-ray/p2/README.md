## How to run the code

### Using `Makefile`

We ship with a `Makefile` that helps to run the code and reproduce the trace. 
We make the following assumption about the running environment:

- We assume the tensorboard is installed at the `$HOME` directiory via the command
`pip install --user tensorflow`. If you have tensorboard installed somewhere else,
then the reproduced traces will not be the same as the provide ones. However, this
won't prevent the `Makefile` from working.
- We assume you have tensorflow installed.
- We assume the shipped file directory structure.
- We assume you run `make setup` before running any commands list below.

 Then you can reproduce the experiments using the following commands:

| Way to construct orthographic feature vector | Add orthographic features at the input | Add orthographic features at the output |
|----------------------------------------------|----------------------------------------|-----------------------------------------|
| embedding                                    | `make i_e`                             | `make o_e`                              |
| one_hot                                      | `make i_o`                             | `make o_o`                              |
| int                                          | `make i_i`                             | `make o_i`                              |

If you perfer to run all the experiments, run `make all`.

__NOTE:__ you can customize the `Makefile` by modify the following variables at the very start of the `Makefile`:

```
PYTHON=python                            # python version
POS_TRAIN_I_E_DIR=pos_train_i_e_dir      # input, embedded training directory
POS_TRAIN_I_O_DIR=pos_train_i_o_dir      # input, one_hot training directory
POS_TRAIN_I_I_DIR=pos_train_i_i_dir      # input, int training directory
POS_TRAIN_O_E_DIR=pos_train_o_e_dir      # output, embedded training directory
POS_TRAIN_O_O_DIR=pos_train_o_o_dir      # output, one_hot training directory
POS_TRAIN_O_I_DIR=pos_train_o_i_dir      # output, int training directory
POS_TRAIN_B_DIR=pos_train_b_dir          # baseline training directory
TRACE_DIR=trace                          # trace directory
SRC_DIR=src                              # src directory location
DATA_DIR=wsj                             # wsj data directory location
```

### Using command line

You can also run the experiment directly through command line. The synopsis of the command looks like below:

    python [src_directory_path]/pos_bilstm.py [data_directory_path] [train_directory_path] standard [train | test] [input | output | none] [embedded | one_hot | int]

Examples:

- python src/pos_bilstm.py wsj pos_train_b_dir standard train none               // Run the baseline model
- python src/pos_bilstm.py wsj pos_train_i_e_dir standard train input embedded   // Run the model with adding orthographic features at the input using embedding
- python src/pos_bilstm.py wsj pos_train_o_o_dir standard train output one_hot   // Run the model with adding orthographic features at the output using one-hot representation

## File Directory Structure

```
.
├── Makefile
├── README.md
├── report.pdf
├── src
│   ├── pos_bilstm.py
│   └── preprocess.py
├── tensorboard_plots
│   ├── baseline
│   │   ├── train_accuracy_loss.png
│   │   ├── valid_OOV.png
│   │   └── valid_accuracy_loss.png
│   ├── input_embedding
│   │   ├── train_accuracy_loss.png
│   │   ├── valid_OOV.png
│   │   └── valid_accuracy_loss.png
│   ├── input_int
│   │   ├── train_accuracy_loss.png
│   │   ├── valid_OOV.png
│   │   └── valid_accuracy_loss.png
│   ├── input_one_hot
│   │   ├── train_accuracy_loss.png
│   │   ├── valid_OOV.png
│   │   └── valid_accuracy_loss.png
│   ├── output_embedding
│   │   ├── train_accuracy_loss.png
│   │   ├── valid_OOV.png
│   │   └── valid_accuracy_loss.png
│   ├── output_int
│   │   ├── train_accuracy_loss.png
│   │   ├── valid_OOV.png
│   │   └── valid_accuracy_loss.png
│   └── output_one_hot
│       ├── train_accuracy_loss.png
│       ├── valid_OOV.png
│       └── valid_accuracy_loss.png
├── trace
│   ├── standard-baseline.txt
│   ├── standard-input-embedded.txt
│   ├── standard-input-int.txt
│   ├── standard-input-one_hot.txt
│   ├── standard-output-embedded.txt
│   ├── standard-output-int.txt
│   └── standard-output-one_hot.txt
└── wsj
```
