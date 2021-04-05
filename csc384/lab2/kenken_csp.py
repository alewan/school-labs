'''
All models need to return a CSP object, and a list of lists of Variable objects 
representing the board. The returned list of lists is used to access the 
solution. 

For example, after these three lines of code

    csp, var_array = kenken_csp_model(board)
    solver = BT(csp)
    solver.bt_search(prop_FC, var_ord)

var_array[0][0].get_assigned_value() should be the correct value in the top left
cell of the KenKen puzzle.

The grid-only models do not need to encode the cage constraints.

1. binary_ne_grid (worth 10/100 marks)
    - A model of a KenKen grid (without cage constraints) built using only 
      binary not-equal constraints for both the row and column constraints.

2. nary_ad_grid (worth 10/100 marks)
    - A model of a KenKen grid (without cage constraints) built using only n-ary 
      all-different constraints for both the row and column constraints. 

3. kenken_csp_model (worth 20/100 marks) 
    - A model built using your choice of (1) binary binary not-equal, or (2) 
      n-ary all-different constraints for the grid.
    - Together with KenKen cage constraints.

'''

from cspbase import *
import itertools


# Create a 2D list of size (end-start) according to requirements (cannot use a 1D list instead)
def create_kenken_board_variables(grid_start, grid_end, var_dom):
    # Generate vars (iterate through whole grid - x is row, y is column)
    kenken_vars = list()
    for x_val in range(grid_start, grid_end):
        row_list = list()
        for y_val in range(grid_start, grid_end):
            row_list.append(Variable('V{}{}'.format(x_val, y_val), var_dom))
        kenken_vars.append(row_list)
    return kenken_vars


# Generate the kenken CSP from a list of vars and a list of constraints
def create_kenken_csp_from_lists(kenken_vars_list, kenken_constraints_list):
    new_kenken_csp = CSP("Kenken")
    for k_var_list in kenken_vars_list:
        for k_var in k_var_list:
            new_kenken_csp.add_var(k_var)
    for k_cons in kenken_constraints_list:
        new_kenken_csp.add_constraint(k_cons)
    return new_kenken_csp


def binary_ne_grid(kenken_grid):
    # Define a grid_limits tuple to access endpoints (in case we make changes to their location in kenken_grid later)
    grid_limits = (1, kenken_grid[0][0])

    # Need a domain list to create variables
    var_domain = list()
    for value in range(grid_limits[0], grid_limits[1] + 1):
        var_domain.append(value)

    # Generate all possible cross values in the domain (valuable for the constraints)
    cross_values = list()
    for element in itertools.permutations(var_domain, 2):
        cross_values.append(element)

    # Generate variables
    kenken_bne_vars = create_kenken_board_variables(grid_limits[0], grid_limits[1] + 1, var_domain)

    # Generate binary constraints
    kenken_bne_constraints = list()
    # Rows
    for x_val in range(grid_limits[1]):
        for y_val in range(grid_limits[1]):
            for k in range(grid_limits[1]):
                if k > y_val:
                    v1 = kenken_bne_vars[x_val][y_val]
                    v2 = kenken_bne_vars[x_val][k]
                    new_cons = Constraint("C(V{}{},V{}{})".format(x_val + 1, y_val + 1, x_val + 1, k + 1), [v1, v2])
                    new_tuple = list()
                    for element in cross_values:
                        new_tuple.append(element)
                    new_cons.add_satisfying_tuples(new_tuple)
                    kenken_bne_constraints.append(new_cons)
    # Columns
    for y_val in range(grid_limits[1]):
        for x_val in range(grid_limits[1]):
            for k in range(grid_limits[1]):
                if k > x_val:
                    v1 = kenken_bne_vars[x_val][y_val]
                    v2 = kenken_bne_vars[k][y_val]
                    new_cons = Constraint("C(V{}{},V{}{})".format(x_val + 1, y_val + 1, k + 1, y_val + 1), [v1, v2])
                    new_tuple = list()
                    for element in cross_values:
                        new_tuple.append(element)
                    new_cons.add_satisfying_tuples(new_tuple)
                    kenken_bne_constraints.append(new_cons)

    # Create the CSP
    kenken_bne_csp = create_kenken_csp_from_lists(kenken_bne_vars, kenken_bne_constraints)

    return kenken_bne_csp, kenken_bne_vars


def nary_ad_grid(kenken_grid):
    # Define a grid_limits tuple to access endpoints (in case we make changes to their location in kenken_grid later)
    grid_limits = (1, kenken_grid[0][0])

    # Need a domain list to create variables
    var_domain = list()
    for value in range(grid_limits[0], grid_limits[1] + 1):
        var_domain.append(value)

    # Generate all possible cross values in the domain (valuable for the constraints)
    cross_values = list()
    for element in itertools.permutations(var_domain, grid_limits[1]):
        cross_values.append(element)

    # Generate variables
    kenken_nary_vars = create_kenken_board_variables(grid_limits[0], grid_limits[1] + 1, var_domain)

    # Generate constraints
    kenken_nary_constraints = list()
    for index in range(grid_limits[1]):
        new_row_list = kenken_nary_vars[index]
        new_cons_row = Constraint("C(Row{})".format(index + 1), new_row_list)
        new_col_list = list()
        for var_lst in kenken_nary_vars:
            new_col_list.append(var_lst[index])
        new_cons_col = Constraint("C(Column{})".format(index + 1), new_col_list)

        new_tuple1 = list()
        new_tuple2 = list()
        for element in cross_values:
            new_tuple1.append(element)
            new_tuple2.append(element)

        new_cons_row.add_satisfying_tuples(new_tuple1)
        new_cons_col.add_satisfying_tuples(new_tuple2)

        kenken_nary_constraints.append(new_cons_row)
        kenken_nary_constraints.append(new_cons_col)

    # Create the CSP
    kenken_nary_csp = create_kenken_csp_from_lists(kenken_nary_vars, kenken_nary_constraints)

    return kenken_nary_csp, kenken_nary_vars


def kenken_csp_model(kenken_grid):
    # Generate the initial grid using the binary model
    kenken_csp, kenken_csp_vars = binary_ne_grid(kenken_grid)

    # Generate cage constraints
    kenken_cage_constraints = list()
    for cage_index in range(1, len(kenken_grid)):
        # Check for a long enough list
        if len(kenken_grid[cage_index]) > 2:
            # Get the first 1, ..., n-2 elements (the vars)
            cage_vars = list()
            cage_vars_domain = list()  # Need to track separately since cage_vars needed for cage constraint
            for cell_in_cage_index in range(len(kenken_grid[cage_index]) - 2):
                cell = str(kenken_grid[cage_index][cell_in_cage_index])
                x, y = int(cell[0]) - 1, int(cell[1]) - 1
                cage_vars.append(kenken_csp_vars[x][y])
                cage_vars_domain.append(kenken_csp_vars[x][y].domain())

            # Create new cons
            new_cage_cons = Constraint("C(Cage{})".format(cage_index), cage_vars)

            # Get op and required value
            op = kenken_grid[cage_index][-1]  # Last element in the list
            target_val = kenken_grid[cage_index][-2]  # Second last element in the list

            # Generate valid tuples
            possible_sat_tuples = list()

            # Decode the operator and generate satisfying tuples
            # For cage_vars_domain use the * operator to unpack the 'list of lists' domain to give the possible lists
            # Addition - check if all the numbers add to the desired value
            if op == 0:
                for possible_tuple in itertools.product(*cage_vars_domain):
                    temp_val = target_val
                    for val in possible_tuple:
                        temp_val -= val
                    if temp_val == 0:
                        possible_sat_tuples.append(possible_tuple)

            # Subtraction - check subtracting numbers results in the desired value
            elif op == 1:
                for possible_tuple in itertools.product(*cage_vars_domain):
                    for possible_val in itertools.permutations(possible_tuple):
                        temp_val = possible_val[0]
                        for temp_ind in range(1, len(possible_val)):
                            temp_val -= possible_val[temp_ind]
                        if temp_val == target_val:
                            possible_sat_tuples.append(possible_tuple)

            # Division, check that repeated division results in the target
            elif op == 2:
                for possible_tuple in itertools.product(*cage_vars_domain):
                    for possible_val in itertools.permutations(possible_tuple):
                        temp_val = possible_val[0]
                        for temp_ind in range(1, len(possible_val)):
                            temp_val /= possible_val[temp_ind]
                        if temp_val == target_val:
                            possible_sat_tuples.append(possible_tuple)

            # Assume we only have valid input - last case is multiplication
            else:
                for possible_tuple in itertools.product(*cage_vars_domain):
                    temp_val = 1
                    for val in possible_tuple:
                        temp_val *= val
                    if temp_val == target_val:
                        possible_sat_tuples.append(possible_tuple)

            # Add the satisfying tuples to the constraint and the constraint to the list
            new_cage_cons.add_satisfying_tuples(possible_sat_tuples)
            kenken_cage_constraints.append(new_cage_cons)

    # Add cage constraints
    for cage_constraint in kenken_cage_constraints:
        kenken_csp.add_constraint(cage_constraint)

    return kenken_csp, kenken_csp_vars
