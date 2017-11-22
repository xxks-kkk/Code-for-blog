import numpy as np
from matplotlib import pyplot as plt
import random
import sys
import logging
import heapq
import re

# Reference: https://docs.python.org/2/howto/logging.html
logging.basicConfig(format='%(message)s', filename='rl.log',filemode='w',level=logging.DEBUG)

REWARD_EMPTY = 0
REWARD_OBJ = 5
REWARD_LITTER = -4
REWARD_EXIT = 100

# Generate the reward matrix for avoiding litters module
def avoidLittersReward(params=None):
    np.random.seed(10)
    random.seed(10)

    numRows = params["numRows"]
    numCols = params["numCols"]
    numLitters = params["numLitters"]
    numObjs = params["numObjs"]

    # Sanity check
    numSlots = numRows * numCols
    assert numSlots >= numObjs + numLitters

    R = np.zeros(shape=(numRows, numCols))
    # used to indicate the state of the grid ("L", "O", "E", "X")
    R_state = np.empty(shape=(numRows, numCols), dtype=str)

    # We add movement reward value -0.1
    R[:,:] = REWARD_EMPTY
    # We initialize all the grid states to "E" (empty)
    R_state[:,:] = "E"

    # We randomly place the litters on the (use random.sample to avoid generate duplicates)
    #litters_x = np.asarray(random.sample(range(numCols), numLitters))
    #litters_y = np.asarray(random.sample(range(numRows), numLitters))
    litters_x = np.random.randint(0,numCols,numLitters)
    litters_y = np.random.randint(0,numRows,numLitters)

    for i,j in zip(litters_x, litters_y):
        R[j,i] = REWARD_LITTER
        R_state[j,i] = "L"

    logging.debug("avoidLittersReward:: R: ")
    logging.debug(R)
    logging.debug("avoidLittersReward:: R_state: ")
    logging.debug(R_state)
    return R, R_state


# Generate the reward matrix for picking up objects module
def pickingObjsReward(params=None):
    np.random.seed(10)
    random.seed(12)

    numRows = params["numRows"]
    numCols = params["numCols"]
    numLitters = params["numLitters"]
    numObjs = params["numObjs"]

    # Sanity check
    numSlots = numRows * numCols
    assert numSlots >= numObjs + numLitters

    R = np.zeros(shape=(numRows, numCols))
    # used to indicate the state of the grid ("L", "O", "E", "X")
    R_state = np.empty(shape=(numRows, numCols), dtype=str)

    # We add movement reward value -0.1
    R[:,:] = REWARD_EMPTY
    # We initialize all the grid states to "E" (empty)
    R_state[:,:] = "E"

    # We randomly place the litters on the map
    #objs_x = np.asarray(random.sample(range(numCols), numObjs))
    #objs_y = np.asarray(random.sample(range(numRows), numObjs))
    objs_x = np.random.randint(0, numCols, numObjs)
    objs_y = np.random.randint(0, numRows, numObjs)

    for i,j in zip(objs_x, objs_y):
        R[j,i] = REWARD_OBJ
        R_state[j,i] = "O"

    logging.debug("pickingObjsReward:: R: ")
    logging.debug(R)
    logging.debug("pickingObjsReward:: R_state: ")
    logging.debug(R_state)
    return R, R_state


# Generate the reward matrix for exiting the map module
def exitingMapReward(params=None):
    np.random.seed(10)

    numRows = params["numRows"]
    numCols = params["numCols"]
    numLitters = params["numLitters"]
    numObjs = params["numObjs"]

    # Sanity check
    numSlots = numRows * numCols
    assert numSlots >= numObjs + numLitters

    R = np.zeros(shape=(numRows, numCols))
    # used to indicate the state of the grid ("L", "O", "E", "X")
    R_state = np.empty(shape=(numRows, numCols), dtype=str)

    # We add movement reward value -0.1
    R[:,:] = REWARD_EMPTY
    # We initialize all the grid states to "E" (empty)
    R_state[:,:] = "E"
    # We add reward to exit
    R[:, -1] = REWARD_EXIT
    R_state[:,-1] = "X"

    logging.debug("exitingMapReward:: R: ")
    logging.debug(R)
    logging.debug("exitingMapReward:: R_state: ")
    logging.debug(R_state)
    return R, R_state


# mode = 0 (litter only); 1 (obj only); 2 (exit only); 3 (all)
# policy = 0 (module_aggregation_algorithm); 1 (module_selection_algorithm); 2 (module_voting_algorithm)
def drawGrid(Q=None, R=None, R_state=None, numSteps=200, mode=0, policy=0, checkRepeatActions=True):
    # Reference: "Global Policy Construction in Modular Reinforcement Learning"
    def module_aggregation_algorithm(s, Q=None):
        Q_all = sum(Q)
        return np.argmax(Q_all[s,:])
    def module_selection_algorithm(s, Q=None):
        W = []
        for Q_elem in Q:
            W.append(np.std(Q_elem[s,:]))
        return np.argmax(Q[np.argmax(W)][s,:])
    def module_voting_algorithm(s, Q=None):
        nActions = np.shape(Q[0])[1]
        K = np.zeros(nActions)
        for Q_elem in Q:
            K[np.argmax(Q_elem[s,:])] += np.std(Q_elem[s,:])
        return np.argmax(K)
    def checkCycle(a):
        # Reference: https://stackoverflow.com/questions/8672853/detecting-a-repeating-cycle-in-a-sequence-of-numbers-python
        # https://stackoverflow.com/questions/3590165/joining-a-list-that-has-integer-values-with-python
        sequence_in_str = ' '.join(map(str, a))
        regex = re.compile(r'(.+ .+)( \1)+')
        match = regex.search(sequence_in_str)
        if match == None:
            return False
        else:
            return True
    def module_selection_algorithm_alternative(s, Q=None):
        # Select the second best action
        # Reference: https://stackoverflow.com/questions/23473723/how-to-find-position-of-the-second-maximum-of-a-list-in-python
        W = []
        for Q_elem in Q:
            W.append(np.std(Q_elem[s,:]))
        fr = Q[np.argmax(W)][s,:]
        print fr
        return heapq.nlargest(2, xrange(len(fr)), key=fr.__getitem__)[-1]

    litters_x, litters_y, objs_x, objs_y, exit_x, exit_y = [],[],[],[],[],[]
    numRows = np.shape(R[0])[0]
    numCols = np.shape(R[0])[1]
    nActions = np.shape(Q[0])[1]

    logging.debug("drawGrid:: R: ")
    logging.debug(R)
    for R_elem in R:
        for i in range(numRows):
            for j in range(numCols):
                if mode == 0 or mode == 3:
                    # add litter
                    if R_elem[i,j] < 0 and R_elem[i,j] == REWARD_LITTER:
                        litters_x.append(j)
                        litters_y.append(i)
                if mode == 1 or mode == 3:
                    # add objs
                    if R_elem[i,j] > 0 and R_elem[i,j] == REWARD_OBJ:
                        objs_x.append(j)
                        objs_y.append(i)

    exit_x = np.ones(numRows) * (numCols - 1)
    exit_y = range(numRows)

    start_x = np.zeros(numRows)
    start_y = range(numRows)

    logging.debug("litters_x: ")
    logging.debug(litters_x)
    logging.debug("litters_y: ")
    logging.debug(litters_y)
    logging.debug("objs_x: ")
    logging.debug(objs_x)
    logging.debug("objs_y: ")
    logging.debug(objs_y)

    # Keep track of the actions so far
    a_list = []
    # Keep track of accumulative reward received
    reward = 0

    step_x, step_y = [], []
    episode = 0
    numEpisode = numSteps
    nStates = np.shape(Q[0])[0]

    s_R = [np.random.randint(numRows), 0]
    step_x.append(s_R[1])
    step_y.append(s_R[0])
    s = s_R[1] + s_R[0]*numCols

    for R_elem in R:
        reward += R_elem[s_R[0], s_R[1]]

    for idx, R_state_elem in enumerate(R_state):
        if R_state_elem[s_R[0], s_R[1]] == "O":
            R_state_elem[s_R[0], s_R[1]] = "E"
            R[idx][s_R[0], s_R[1]] = REWARD_EMPTY
        elif R_state_elem[s_R[0], s_R[1]] == "L":
            R_state_elem[s_R[0], s_R[1]] = "E"
            R[idx][s_R[0], s_R[1]] = REWARD_EMPTY

    while (episode < numEpisode) and (s % numCols !=  (numCols - 1)) and (s / numCols < numRows):
        logging.debug("drawGrid:: s: " + str(s))
        logging.debug("drawGrid:: s coordinate: " + "[" + str(s / numCols) + "," + str(s % numCols) + "]")

        # We want to keep track of previous locations to see if we are in a loop. If so, we may want to
        # break the loop by randomly pick an action.
        if checkCycle(a_list) and checkRepeatActions:
            #a = module_selection_algorithm_alternative(s, Q)
            a = np.random.randint(nActions)
        else:
            if policy == 0:
                a = module_aggregation_algorithm(s, Q)
            elif policy == 1:
                a = module_selection_algorithm(s,Q)
            elif policy == 2:
                a = module_voting_algorithm(s,Q)

        logging.debug("action: " + str(a))

        a_list.append(a)
        logging.debug("action_list: ")
        logging.debug(a_list)

        if a == 0:  # up
            sprime = s - numCols if (s - numCols) >= 0 else s
        elif a == 1:  # down
            sprime = s + numCols if (s + numCols) < nStates else s
        elif a == 2:  # forward
            sprime = s + 1 if (s % numCols + 1) < numCols else s

        #TODO: calculate the reward we collect so far during one episode and see if we can have reward increase after we iteratively learn
        sprime_row = sprime / numCols
        sprime_col = sprime % numCols
        step_x.append(sprime_col)
        step_y.append(sprime_row)

        for R_elem in R:
            reward += R_elem[sprime_row, sprime_col]

        for idx, R_state_elem in enumerate(R_state):
            if R_state_elem[sprime_row, sprime_col] == "O":
                R_state_elem[sprime_row, sprime_col] = "E"
                R[idx][sprime_row, sprime_col] = REWARD_EMPTY
            elif R_state_elem[sprime_row, sprime_col] == "L":
                R_state_elem[sprime_row, sprime_col] = "E"
                R[idx][sprime_row, sprime_col] = REWARD_EMPTY

        s = sprime
        episode += 1
        print "episode: " + str(episode)
        logging.debug("drawGrid:: sprime: " + str(sprime))
        logging.debug("drawGrid:: sprime coordinate: " + "[" + str(s / numCols) + "," + str(s % numCols) + "]")

    logging.debug("step_x: ")
    logging.debug(step_x)
    logging.debug("step_y: ")
    logging.debug(step_y)

    # Plot the grid
    fig, ax = plt.subplots()
    if mode == 0 or mode == 3:
        litters_plot = plt.scatter(litters_x, litters_y, marker='^', color='red')
        starts_plot = plt.scatter(start_x, start_y, marker='x', color='green')
    if mode == 1 or mode == 3:
        objs_plot = plt.scatter(objs_x, objs_y, marker='o', color='orange')
        starts_plot = plt.scatter(start_x, start_y, marker='x', color='green')
    if mode == 2 or mode == 3:
        exit_plot = plt.scatter(exit_x, exit_y, marker='x', color='green')
        starts_plot = plt.scatter(start_x, start_y, marker='x', color='green')
    plt.plot(step_x, step_y)
    if mode == 0:
        legend_plot = [litters_plot, starts_plot]
        legend_text = ["Litters", "Starts (left)"]
    if mode == 1:
        legend_plot = [objs_plot, starts_plot]
        legend_text = ["Objects", "Starts (left)"]
    if mode == 2:
        legend_plot = [exit_plot, starts_plot]
        legend_text = ["Exits", "Starts (left)"]
    if mode == 3:
        legend_plot = (litters_plot, objs_plot, starts_plot, exit_plot)
        legend_text = ('Litters', 'Objects', 'Starts (left)', 'Exits (right)')
    if mode == 0 or mode == 1 or mode == 2 or mode == 3:
        plt.legend(legend_plot,
                   legend_text,
                   scatterpoints=1,
                   loc='upper center',
                   ncol=10,
                   fontsize=8,
                   fancybox = True,
                   shadow = True,
                   bbox_to_anchor=(0.5, -0.05))
    plt.gca().invert_yaxis()
    ax.set_ylabel('start')
    if mode == 0:
        plt.title("Avoiding litters module with total reward: " + str(reward))
    if mode == 1:
        plt.title("Picking up objects module with total reward: " + str(reward))
    if mode == 2:
        plt.title("Walking module with total reward: " + str(reward))
    if mode == 3:
        plt.title("All modules combined with total reward: " + str(reward))

    # Shrink current axis's height by 10% on the bottom
    # Reference: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    plt.show()


# SARSA algorithm
# Reference: - http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html
#            - http://mnemstudio.org/path-finding-q-learning-tutorial.htm
def sarsa(R, R_state, hyperparams=None):
    numRows = np.shape(R)[0]
    numCols = np.shape(R)[1]
    nStates = numRows * numCols
    # We only support three actions: up, down, forward
    nActions = 3
    Q = np.random.rand(nStates, nActions)*0.1-0.05
    R_bk = np.copy(R)
    R_state_bk = np.copy(R_state)

    step = 0
    episode = 0

    mu = hyperparams["mu"]
    gamma = hyperparams["gamma"] # discount factor
    epsilon = hyperparams["epsilon"]
    numEpisode = hyperparams["numEpisode"]
    numSteps = hyperparams["numSteps"]

    reward = 0

    while episode < numEpisode:
        logging.debug("episode: " + str(episode))
        # pick initial state from the start column (i.e, 1st column). Store the result in terms of R matrix index
        s_R = [np.random.randint(numRows), 0]
        logging.debug("s_R: ")
        logging.debug(s_R)
        # we map the initial state to the Q matrix; Q matrix is organized in row order of the R matrix
        s = s_R[1] + s_R[0]*numCols
        # epsilon-greedy
        if (np.random.rand() < epsilon):
            a = np.random.randint(nActions)
        else:
            a = np.argmax(Q[s, :])

        while (s % numCols !=  (numCols - 1)) and step < numSteps:
            logging.debug("step: " + str(step))
            logging.debug("s: " + str(s))
            logging.debug("s coordinate: " + "[" + str(s / numCols) + "," + str(s % numCols) + "]")
            logging.debug("a: " + str(a))
            if a == 0: # up
                logging.debug("up")
                sprime = s - numCols if (s - numCols) >= 0 else s
                sprime_row = sprime / numCols
                sprime_col = sprime % numCols
                r = R_bk[ sprime_row, sprime_col]
                if R_state_bk[sprime_row, sprime_col] == "O":
                    R_state_bk[sprime_row, sprime_col] = "E"
                    R_bk[sprime_row, sprime_col] = REWARD_EMPTY
                elif R_state_bk[sprime_row, sprime_col] == "L":
                    R_state_bk[sprime_row, sprime_col] = "E"
                    R_bk[sprime_row, sprime_col] = REWARD_EMPTY
            elif a == 1: # down
                logging.debug("down")
                sprime = s + numCols if (s + numCols) < nStates else s
                sprime_row = sprime / numCols
                sprime_col = sprime % numCols
                r = R_bk[sprime_row, sprime_col]
                if R_state_bk[sprime_row, sprime_col] == "O":
                    R_state_bk[sprime_row, sprime_col] = "E"
                    R_bk[sprime_row, sprime_col] = REWARD_EMPTY
                elif R_state_bk[sprime_row, sprime_col] == "L":
                    R_state_bk[sprime_row, sprime_col] = "E"
                    R_bk[sprime_row, sprime_col] = REWARD_EMPTY
            elif a == 2: # forward
                logging.debug("forward")
                sprime = s + 1 if (s % numCols + 1) < numCols else s
                sprime_row = sprime / numCols
                sprime_col = sprime % numCols
                r = R_bk[sprime_row, sprime_col]
                if R_state_bk[sprime_row, sprime_col] == "O":
                    R_state_bk[sprime_row, sprime_col] = "E"
                    R_bk[sprime_row, sprime_col] = REWARD_EMPTY
                elif R_state_bk[sprime_row, sprime_col] == "L":
                    R_state_bk[sprime_row, sprime_col] = "E"
                    R_bk[sprime_row, sprime_col] = REWARD_EMPTY

            # epsilon-greedy
            if (np.random.rand() < epsilon):
                aprime = np.random.randint(nActions)
            else:
                aprime = np.argmax(Q[sprime, :])
            Q[s,a] += mu * (r + gamma*Q[sprime, aprime] - Q[s,a])

            logging.debug("sprime coordinate: " + "[" + str(sprime / numCols) + "," + str(sprime % numCols) + "]")
            s = sprime
            a = aprime
            step += 1

            reward += r

        print("episode: " + str(episode) + "reward: " + str(reward))
        R_bk = np.copy(R)
        R_state_bk = np.copy(R_state)
        episode += 1
        step = 0
        reward = 0

    logging.debug("Q:")
    logging.debug(Q)
    return Q

if __name__ == "__main__":

    params = {
        "numRows": 100,
        "numCols": 100,
        "numLitters": 30,
        "numObjs": 40
    }

    R_litter, R_state_litter = avoidLittersReward(params)
    R_objs, R_state_objs = pickingObjsReward(params)
    R_exit, R_state_exit = exitingMapReward(params)

    litter_hyperparams = {
        "mu": 0.7,
        "gamma": 0.4,
        "epsilon": 0.5,
        "numEpisode": 200,
        "numSteps": 100
    }

    logging.debug("=" * 10)
    logging.debug("Litter SARSA")
    Q_litter = sarsa(R_litter, R_state_litter, hyperparams=litter_hyperparams)

    objs_hyperparams = {
        "mu": 0.7,
        "gamma": 0.7,
        "epsilon": 0.5,
        "numEpisode": 200,
        "numSteps": 100
    }

    logging.debug("=" * 10)
    logging.debug("Objs SARSA")
    Q_objs = sarsa(R_objs, R_state_objs, hyperparams=objs_hyperparams)

    exit_hyperparams = {
        "mu": 0.7,
        "gamma": 1,
        "epsilon": 0.5,
        "numEpisode": 200,
        "numSteps": 100
    }

    logging.debug("=" * 10)
    logging.debug("Exits SARSA")
    Q_exit = sarsa(R_exit, R_state_exit, hyperparams=exit_hyperparams)

    R_all = R_objs + R_exit + R_litter

    logging.debug(R_objs)
    logging.debug(R_exit)
    logging.debug(R_litter)
    logging.debug(R_all)

    if len(sys.argv) >= 2:
        system_to_run = sys.argv[1]
    else:
        system_to_run = "A"
    if system_to_run == "L":
        drawGrid([Q_litter], [R_litter], R_state=[R_state_litter], mode=0, policy=1, checkRepeatActions=False)
    if system_to_run == "O":
        drawGrid([Q_objs], [R_objs], R_state=[R_state_objs], mode=1, policy=0, checkRepeatActions=False)
    if system_to_run == "X":
        drawGrid([Q_exit], [R_exit], R_state=[R_state_exit], mode=2, policy=0, checkRepeatActions=False)
    if system_to_run == "A":
        drawGrid([Q_litter, Q_objs, Q_exit], [R_objs, R_exit, R_litter], numSteps=1000, R_state=[R_state_objs, R_state_exit, R_state_litter], checkRepeatActions=True, mode=3, policy=2)
