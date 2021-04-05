'''
This file will contain different constraint propagators to be used within 
bt_search.

---
A propagator is a function with the following header
    propagator(csp, newly_instantiated_variable=None)

csp is a CSP object---the propagator can use this to get access to the variables 
and constraints of the problem. The assigned variables can be accessed via 
methods, the values assigned can also be accessed.

newly_instantiated_variable is an optional argument. SEE ``PROCESSING REQUIRED''
if newly_instantiated_variable is not None:
    then newly_instantiated_variable is the most
    recently assigned variable of the search.
else:
    propagator is called before any assignments are made
    in which case it must decide what processing to do
    prior to any variables being assigned. 

The propagator returns True/False and a list of (Variable, Value) pairs, like so
    (True/False, [(Variable, Value), (Variable, Value) ...]

Propagators will return False if they detect a dead-end. In this case, bt_search 
will backtrack. Propagators will return true if we can continue.

The list of variable value pairs are all of the values that the propagator 
pruned (using the variable's prune_value method). bt_search NEEDS to know this 
in order to correctly restore these values when it undoes a variable assignment.

Propagators SHOULD NOT prune a value that has already been pruned! Nor should 
they prune a value twice.

---

PROCESSING REQUIRED:
When a propagator is called with newly_instantiated_variable = None:

1. For plain backtracking (where we only check fully instantiated constraints)
we do nothing...return true, []

2. For FC (where we only check constraints with one remaining 
variable) we look for unary constraints of the csp (constraints whose scope 
contains only one variable) and we forward_check these constraints.

3. For GAC we initialize the GAC queue with all constaints of the csp.

When a propagator is called with newly_instantiated_variable = a variable V

1. For plain backtracking we check all constraints with V (see csp method
get_cons_with_var) that are fully assigned.

2. For forward checking we forward check all constraints with V that have one 
unassigned variable left

3. For GAC we initialize the GAC queue with all constraints containing V.

'''


# Helper function to add a new value to a list if it doesn't exist
def add_unique(new_val, lst):
    if new_val not in lst:
        lst.append(new_val)
    return


# Helper function to add new values from a list to another list
def add_unique_from_list(src, dest):
    for element in src:
        if element not in dest:
            dest.append(element)
    return


def prop_BT(csp, newVar=None):
    '''
    Do plain backtracking propagation. That is, do no propagation at all. Just 
    check fully instantiated constraints.
    '''
    if not newVar:
        return True, []
    for c in csp.get_cons_with_var(newVar):
        if c.get_n_unasgn() == 0:
            vals = []
            vars = c.get_scope()
            for var in vars:
                vals.append(var.get_assigned_value())
            if not c.check(vals):
                return False, []
    return True, []


# Forward Checking (FC) [Reference: CSP Slides 50/51]
def prop_FC(csp, newVar=None):
    # Set up lists for constraints and pruned values
    # Check if new var has a value (otherwise check all as per lab specs)
    cons = csp.get_cons_with_var(newVar) if newVar else csp.get_all_cons()
    curr_pruned_assignments = list()

    # Iterate through constraints
    for c in cons:
        # Check if only one var unassigned (as we are doing FC)
        if c.get_n_unasgn() == 1:
            # Get the unassigned var
            unassigned_var = c.get_unasgn_vars()[0]

            # Iterate over possible assignments in the domain
            for possible_val in unassigned_var.cur_domain():
                # If assignment doesn't satisfy constraint, prune it
                if not c.has_support(unassigned_var, possible_val):
                    add_unique((unassigned_var, possible_val), curr_pruned_assignments)
                    unassigned_var.prune_value(possible_val)

            # Check for DWO, if so return false to indicate the DWO and send list of assignments that were pruned
            if not unassigned_var.cur_domain_size():
                return False, curr_pruned_assignments

    # Finished checking without a DWO, return the list we of assignments pruned
    return True, curr_pruned_assignments


# Generalized Arc Constraint (GAC) [Reference: CSP Slides 88/89]
def prop_GAC(csp, newVar=None):
    '''
    Do GAC propagation. If newVar is None we do initial GAC enforce processing 
    all constraints. Otherwise we do GAC enforce with constraints containing 
    newVar on GAC Queue.
    '''
    # Set up list for pruned values
    curr_pruned_assignments = list()

    # Fill the GAC Queue by checking if new var has a value (otherwise check all as per lab specs)
    gac_queue = (csp.get_cons_with_var(newVar) if newVar else csp.get_all_cons())

    # Keep iterating until the GAC Queue is empty
    while gac_queue:
        # Get the first constraint from the GAC Queue
        cons = gac_queue.pop(0)

        # Iterate through variables in the constraint
        for var in cons.get_unasgn_vars():
            # Iterate over possible assignments in the domain
            for possible_val in var.cur_domain():
                # If assignment doesn't satisfy constraint, prune it similar to FC
                if not cons.has_support(var, possible_val):
                    if (var, possible_val) not in curr_pruned_assignments:
                        var.prune_value(possible_val)
                        curr_pruned_assignments.append((var, possible_val))

                    # Check for DWO, if so return false to indicate DWO and send list of assignments that were pruned
                    # Otherwise add the constraints that include the variable we are iterating over to the GAC queue
                    if var.cur_domain_size():
                        add_unique_from_list(csp.get_cons_with_var(var), gac_queue)
                    else:
                        gac_queue.clear()  # Done to be consistent with alg from slides, but this shouldn't be necessary
                        return False, curr_pruned_assignments

    # Finished checking without a DWO, return the list we of assignments pruned
    return True, curr_pruned_assignments
