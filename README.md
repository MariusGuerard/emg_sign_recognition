# sEMG Sign Recognition.

This project is called somOS.
My friend and I built from scratch a sEMG armband that allows you to interact via bluetooth with an Android smartphone.

As a first step, you can record static signs that are linked with different actions (moving a cursor and clicking, sign recognition for command based personal assistants, ...). This notebook is used as an offline validation of our algorithms. In particular, we wanted to evaluate the difference in performance (measured as the accuracy of sign recognition) between three different situations where we change the conditions of training (when user record signs) and testing (when the app guess what sign the user is doing):

- 1) User record signs and the testing happens right after.
- 2) USer record signs, remove the armband and reposition it at the same place and then test.
- 3) User record signs, give the armband to another user and the app try to recognize the sign of the other user (same signs as the one recorded).

This allow to test the importance of different mode of calibration for an operational use of this armband where people might remove it and where we would like to use data from all our users to improve the algorithms (via transfert learning).


## 1. Data and experiment description.
As the data is light, it is also included with this repo for convenience.
The data are stored in the 'data/data_somos/JAN20/' directory. It is made of 24 text files in csv format:
- 12 files for user 1 named 'mg' (it's me)
--- 2 sessions 's1' and 's2'

- 12 files for user 2 named 'rr'.
--- 2 sessions 's1' and 's2'

Each of the session is composed of 6 samples ('e1', ..., 'e6'). In each of these samples, 4 signs were recorded (Rest, Paper, Rock, and Scissors).

Between each of these 6 samples, the users are not moving the armband. But the armband is removed between each session.


## 2. Code Usage.

### 2.1 Online.
You can run the entire notebook interactively and change the parameters yourself by clicking on this link (also at the beginning of the notebook):

https://colab.research.google.com/github/MariusGuerard/emg_sign_recognition/blob/master/somos_data_analysis.ipynb

and changing the variable 'RUN_IN_COLAB' to true at the beginning of the notebook.

### 2.2 On your Local machine.
You have to clone this repo and install the necessary package listed in the requirement.txt (you can use pip install -r requirement.txt in your virtual environment). 

## 3. Next Steps:

- Try other parameters (N_STEPS, ...), models (SVM, other DNN architecture,...), and metrics, notably with geomstats package.

- Compare with results obtained on MYO armband for similar signs (there is already a Paper, Rock, Scissors available on kaggle).

- Use transfert learning (use the model trained on other users for calibrating the model inter-sessions/intra-sessions) and see if it helps imrpoving accuracy.

- Adding more signs and visalizations

- To compare offline and online algorithms accuracy: online accuracy are usually higher than offline (for same parameters'values) because of the adaptation of the user to the algorithm.

- Implement dynamic sign recognition (for example with NLP + CNN) so that specific movements can be recognized and not only static signs.
