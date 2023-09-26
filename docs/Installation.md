First, clone the repository and download it to your computer. 

Extract the repository code to the desired directory.

add to PYTHONPATH the parent directory of this folder; e.g., `export PYTHONPATH="$PYTHONPATH:/parent/directory/of_ai4netmon/"
To do that in ubuntu, open ~/.bashrc, and write the command above. Then save and close the file. 

In the code directory:
* install requirements.txt; e.g.,
	- create virtual environment; e.g., on Linux
		- `python3 -m venv /path/to/new/virtual/environment`
		- `source /path/to/new/virtual/environment/bin activate`
	- install requirements `pip install -r requirements.txt`
