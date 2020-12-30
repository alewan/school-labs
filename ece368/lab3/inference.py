import numpy as np
import graphics
import rover


def print_non_zero(dist: rover.Distribution):
    for k, v in dist.items():
        if v > 0.0 or v < 0.0:
            print(str(k) + ':', v)
    return


def error_prob(l1, l2):
    """
    :param l1: first list
    :param l2: second list
    :return: Error probability of the list
    """
    return 1 - sum([1 if l1[i] == l2[i] else 0 for i in range(100)]) / 100


def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [rover.Distribution()] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [rover.Distribution()] * num_time_steps
    marginals = [rover.Distribution()] * num_time_steps

    # Cache some data to make things faster
    next_states = {s: transition_model(s) for s in all_possible_hidden_states}
    obs_probs = {s: observation_model(s) for s in all_possible_hidden_states}

    # Do Initialization
    # Refine prior with observation
    for state in prior_distribution:
        forward_messages[0][state] *= obs_probs[state][observations[0]]

    # Calculate the last message first based on moving back from the probability of the final observation
    c = rover.Distribution({s: obs_probs[s][observations[-1]] for s in all_possible_hidden_states})
    c.renormalize()
    backward_messages[-1] = c

    # Compute the forward messages
    for i in range(1, num_time_steps):
        # Iterate through all states in the previous msg, calculate new state probs & sum over all possible transitions
        c = rover.Distribution()
        for old_state, old_state_prob in forward_messages[i - 1].items():
            for potential_state, transition_prob in next_states[old_state].items():
                c[potential_state] = c[potential_state] + old_state_prob * transition_prob
        # Account for the likelihood of the observation given the states calculated from the previous message
        for state, prob in c.items():
            c[state] = prob * obs_probs[state][observations[i]] if observations[i] is not None else prob
        # Renormalize to avoid underflow problem
        c.renormalize()
        forward_messages[i] = c

    # Compute the backward messages, analogous to fwd messages case but in reverse
    for i in range(num_time_steps - 2, -1, -1):
        c = rover.Distribution()
        for state in all_possible_hidden_states:
            for future_state, future_prob in backward_messages[i + 1].items():
                c[state] = c[state] + next_states[state][future_state] * future_prob
        for state, prob in c.items():
            c[state] = prob * obs_probs[state][observations[i]] if observations[i] is not None else prob
        c.renormalize()
        backward_messages[i] = c

    # Compute the marginals
    for i in range(0, num_time_steps):
        # Take the product of the forward & backward messages for all states.
        c = rover.Distribution(
            {state: forward_messages[i][state] * backward_messages[i][state] for state in all_possible_hidden_states})
        # Here we can cheat a little and simply renormalize to obtain the correct probability distribution
        c.renormalize()
        marginals[i] = c

    return marginals


def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of estimated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """
    # NOTE: The code below makes use of the fact that np.log(0) = -inf, which has the property c + (-inf) = -inf and
    # c > -inf for all real numbers. This could have been avoided by checking the argument of the log first.
    # All sections that include this are commented with 'May produce warning'

    # Cache some data to make things faster
    next_states = {s: transition_model(s) for s in all_possible_hidden_states}
    obs_probs = {s: observation_model(s) for s in all_possible_hidden_states}

    num_time_steps = len(observations)
    forward_messages = [rover.Distribution()] * num_time_steps
    fwd_states = [{}] * (num_time_steps - 1)
    estimated_hidden_states = [None] * num_time_steps

    # Forward Initialization using prior adjusted with observation
    forward_messages[0] = prior_distribution
    for state in forward_messages[0]:
        forward_messages[0][state] = np.log(forward_messages[0][state])  # May produce warnings of log(0)
        if observations[0] is not None:
            if obs_probs[state][observations[0]] > 0:
                forward_messages[0][state] += np.log(obs_probs[state][observations[0]])
            else:
                forward_messages[0][state] = 0

    # Compute the forward messages
    for i in range(1, num_time_steps):
        c = rover.Distribution()
        d = {}
        # Iterate through states in the previous msg, calculate new state probs & save argmaxes
        for old_state, old_state_prob in forward_messages[i - 1].items():
            if old_state_prob == 0:
                continue  # The old state isn't actually possible so don't bother calculating its transitions
            for potential_state, transition_prob in next_states[old_state].items():
                v = old_state_prob + np.log(transition_prob)
                if observations[i] is not None:
                    v += np.log(obs_probs[potential_state][observations[i]])  # May produce warnings of log(0)
                # Set the value for the state to the max and save the state that generated it
                if c[potential_state] == 0 or v > c[potential_state]:  # c[potential_state] == 0 checks if uninitialized
                    c[potential_state] = v
                    d[potential_state] = old_state
        fwd_states[i-1] = d
        forward_messages[i] = c

    # Initialize backwards pass by taking argmax of last state (can't use mode)
    max_state, max_prob = None, -1e5
    for state, prob in forward_messages[-1].items():
        if prob != 0 and prob > max_prob:
            max_state = state
            max_prob = prob
    estimated_hidden_states[-1] = max_state

    # Trace backwards by finding the argmax of the next state in the sequence
    for i in range(num_time_steps - 2, -1, -1):
        estimated_hidden_states[i] = fwd_states[i][estimated_hidden_states[i + 1]]

    return estimated_hidden_states


if __name__ == '__main__':

    enable_graphics = True

    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'

    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution = rover.initial_distribution()

    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')

    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    # Added Code #
    #####################################
    # Calculate sequence from marginals
    marginal_based_estimated_states = [e.get_mode() for e in marginals]

    # Part 4
    print('Error probability of Viterbi-based:', error_prob(estimated_states, hidden_states))
    print('Error probability of marginal-based:', error_prob(marginal_based_estimated_states, hidden_states))

    # Part 5
    t = {s: rover.transition_model(s) for s in all_possible_hidden_states}
    found = False
    for i in range(len(marginal_based_estimated_states) - 1):
        if marginal_based_estimated_states[i + 1] not in t[marginal_based_estimated_states[i]]:
            print('Segment broke the marginal-based sequence.')
            print('Index:', i, '| Values:', marginal_based_estimated_states[i], marginal_based_estimated_states[i+1])
            found = True
            break
    if not found:
        print('No segment breaks breaks the marginal-based sequence.')

    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
