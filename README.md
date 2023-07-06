# Structure optimized DMs

This code allows to run [DM_phase](https://github.com/danielemichilli/DM_phase) interactively in a notebook. 

# Example setup

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

### Note

I restructured the DM_phase code, made it Python3 compliant, and thinned it out. The package also includes plenty of other functions. Parts of the fitting code comes from [Leon Oostrum](https://github.com/loostrum). I indicated where appropriate who did what. Code in `extern/psrpy` is a slightly modified version of files from [Presto](https://github.com/scottransom/presto). Code in `extern/time_domain_astronomy_sandbox` is taken from [time_domain_astronomy_sandbox](https://github.com/macrocosme/time_domain_astronomy_sandbox). 

### Note

No license yet selected.
