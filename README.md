# Encare Synthetic Data Hackathon

This goal of this hackathon is to create new synthetic data of medical records.

Create a fork and make the fork private!

## Project Structure
- `/data`: Place your raw `synthetic-data-hackaton-sample.csv` here.
- `/results`: Synthetic outputs will be saved here with timestamps.
- `/examples`: Baseline generators (e.g., Random Sampler).
- `data_processor.py`: Cleaning and imputation logic.
- `data_validator.py`: Function that validates the submission locally before submitting to the platform. 
- `evaluator.py`: Statistical (KS-test) and clinical validation.

## Setup Instructions

### 1. Install Visual Studio Code (VS Code)
Download the installer from code.visualstudio.com.

Run the installer and follow the instructions.

Once open, go to the Extensions view (square icon on the left) and search for "Python" (by Microsoft) and click Install.

### 2. Install Python
### Windows
Download the installer from python.org.

IMPORTANT: Check the box "Add Python to PATH" at the start of the installation. If you miss this, the python command won't work in your terminal.

Verify in PowerShell: python --version

### macOS
macOS comes with an older version of Python. Install the latest version using Homebrew: brew install python or download the .pkg from python.org.

Verify in Terminal: python3 --version (Note: You usually must use python3, not python).

### Linux (Ubuntu/Debian)
Update your package manager: sudo apt update

Install Python:
```bash
sudo apt install python3 python3-venv python3-pip
```
Verify: python3 --version

### 1. Create a Virtual Environment
Open your terminal in the project folder and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

Ensure your csv file is in the /data folder and run main.py

The repo also contains some validation for sense checks. The generated data passing these tests shall not be seen as an indicator for a high score.
