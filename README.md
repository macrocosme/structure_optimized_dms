```shell
# Get code
git clone https://github.com/macrocosme/structure_optimized_dms.git
cd structure_optimized_dms

# (Install virtualenv if necessary)
pip3 install virtualenv

# Create a virtual environment coined `topology-env` for the project
virtualenv env

# Activate the environment
source env/bin/activate

# Install required packages
pip3 install -r requirements.txt

# Make script executable
jupyter notebook

# Deactivate when done working on this project to return to global settings
deactivate
```
