# Environment Setup
You only need to do steps 1 and 2 per machine.

## Step 1

**macOS / Linux**
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

**Windows -- PowerShell**
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows -- cmd.exe**
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

## Step 2
To make the Jupyter kernel use the activated environment, do:

**macOS / Linux**
```bash
python3 -m ipykernel install --user --name PracticalAssignment1 --display-name "Python (Practical Assignment 1)"
```

**Windows -- cmd.exe**
```bash
python -m ipykernel install --user --name PracticalAssignment1 --display-name "Python (Practical Assignment 1)"
```

and select `Python (Practical Assignment 1)` as the selected kernel.

## Step 3
For every time you want to access the environment after doing steps 1 and 2, from the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
jupyter-labs # if you want to use your own notebook software, that also works
```

**Windows**
```powershell
venv\Scripts\activate
jupyter-labs # if you want to use your own notebook software, that also works
```