1. Create a directory "MathSolver"
2. Move the "requirements.txt" file to the newly created directory
3. Install Anaconda/Miniconda from https://www.anaconda.com/download/success
4. Open the directory in Terminal
5. Run the following commands: <br>
   a. `conda create -n MathSolver python=3.10.15 anaconda` <br>
   b. `conda activate MathSolver` <br>
   c. `python --version` <br>
   d. `pip install -r requirements.txt` <br>
   e. `pip install .\tensorflow-2.9.0-cp310-cp310-win_amd64.whl` (Make sure you have the .whl file in the same directory from https://pypi.org/project/tensorflow/2.9.0/#files)
7. Run `python app.py` to start up the application

YouTube Tutorial: https://youtu.be/B2L9lrE5WQg
