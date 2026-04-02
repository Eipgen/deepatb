# DeePaTB

## Install Manual 

### Requirements
First, you need to install Amesp. You can download it from the official website: https://www.amesp.xyz/download/. We also provide a compatible version of Amesp for your convenience.  
We **strongly recommend** using `python==3.9.0` to ensure optimal compatibility.

#### Step 1: Configure Amesp and Python environment
```bash
# Add Amesp binary directory to system PATH
export PATH=$PATH:/path/to/amesp/bin/

# Create a dedicated Conda environment with Python 3.9.0
mamba create -n deepatb2 python=3.9.0 -y

# Activate the environment (critical step)
conda activate deepatb2

# Install Python dependencies
pip install -r requirements.txt

# Alternative: Install with Tsinghua PyPI mirror for faster download (China mainland)
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```
#### Step 2: Install modified DeePKS-kit
cd deepks-kit
```
cd deepks-kit
python setup.py install
```

## Usage
A complete example job for Qm7BT is provided in the job directory. Follow the steps below to run the workflow:

#### Step 1: Prepare training datasets
```
cd job/01_prepare

# Generate Amesp input files (.aip) - charge and spin are set to 0 and 1 for all systems
python 00_xyzaip.py

# Run Amesp calculations - ensure all jobs complete successfully
sh 01_run.sh

# Generate atom.npy file in npydata directory
python 02_xyztoatomnpy.py --dir file

# Generate descriptor (dm_eig) in npydata directory
python 03_get_aTB_decriptor.py --dir file

# Generate energy label file
python 04_get_delta_energy.py
```
#### Step 1: Train the model

```
cd ../02_train  # Navigate to training directory

# Start model training
sh train.sh

# Calculate predicted energies using the trained DeePaTB model
python get_deepatb_energy.py
```