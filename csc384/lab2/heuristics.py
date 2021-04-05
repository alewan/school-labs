'''
This file will contain different variable ordering heuristics to be used within
bt_search.

1. ord_dh(csp)
    - Takes in a CSP object (csp).
    - Returns the next Variable to be assigned as per the DH heuristic.
2. ord_mrv(csp)
    - Takes in a CSP object (csp).
    - Returns the next Variable to be assigned as per the MRV heuristic.
3. val_lcv(csp, var)
    - Takes in a CSP object (csp), and a Variable object (var)
    - Returns a list of all of var's potential values, ordered from best value 
      choice to worst value choice according to the LCV heuristic.

The heuristics can use the csp argument (CSP object) to get access to the 
variables and constraints of the problem. The assigned variables and values can 
be accessed via methods.
'''

import random
from copy import deepcopy


# This heuristic just looks at the variable involved in the most constraints on other variables (Degree Heuristic)
def ord_dh(csp):
    unassigned_vars = csp.get_all_unasgn_vars()
    if not unassigned_vars:
        return None
    max_cons = 0
    curr_max_var = unassigned_vars[0]

    for var in unassigned_vars:
        curr_cons = 0
        for cons in csp.get_cons_with_var(var):
            if cons.get_n_unasgn() > 1:
                curr_cons += 1
        if curr_cons > max_cons:
            max_cons = curr_cons
            curr_max_var = var

    return curr_max_var


# This heuristic just finds the most constrained variable (i.e. the one with the minimum remaining values - MRV)
def ord_mrv(csp):
    unassigned_vars = csp.get_all_unasgn_vars()
    if not unassigned_vars:
        return None
    most_constrained = unassigned_vars[0]
    smallest_domain_size = unassigned_vars[0].cur_domain_size()

    for var in unassigned_vars:
        if var.cur_domain_size() < smallest_domain_size:
            most_constrained = var
            smallest_domain_size = var.cur_domain_size()

    return most_constrained


# Least Constraining Value Heuristic
def val_lcv(csp, var):
    # Create data structures, get the list of constraints with the variable
    result = list()
    related_constraints = csp.get_cons_with_var(var)

    # Go through all possible variable assignments in the current domain
    for val in var.cur_domain():
        # Temporarily assign the value and create a temp counter
        var.assign(val)
        tmp_counter = 0

        # Check every variable in every related constraint to examine the impact of the change
        # This could probably be done more efficiently
        for related_constraint in related_constraints:
            for related_var in related_constraint.get_unasgn_vaars():
                for related_var_val in related_var.cur_domain():
                    if not related_constraint.has_support(related_var, related_var_val):
                        tmp_counter += 1

        # Take note of the ruled out values due to the assignment and reset the var
        result.append((val, tmp_counter))
        var.unassign()

    # Sort the list in ascending order based on the counter value and copy the values to a new list
    result.sort(key=lambda res: res[1])
    return_list = list()
    for element in result:
        return_list.append(element[0])

    return return_list
