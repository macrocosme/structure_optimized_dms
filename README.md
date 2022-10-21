# Structure optimized DMs

This code includes allows to run [DM_phase](https://github.com/danielemichilli/DM_phase) interactively in a notebook. 

### Note

I restructured the DM_phase code, made it Python3 compliant, and thinned it out. The package also include plenty of other functions. Parts of the fitting code comes from Leon Oostrum. I indicated where appropriate who did what. I do not know where all the code in `extern/psrpy` comes from. I think some is taken from [Preso](https://github.com/scottransom/presto). If you recognise this code, please contact me so I can acknowledge appropriately. 


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

# Start notebook
jupyter notebook

# Deactivate when done working on this project to return to global settings
deactivate
```
