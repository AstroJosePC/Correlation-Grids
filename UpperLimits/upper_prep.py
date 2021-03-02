# Now make a function to generate the appropriate marker and size dictionaries!?
# first define a function to identify limits:

def upp_mask(data, parameter: str, parameter_err: str):
    return (data[parameter_err] > data[parameter]).values


# second define a function to map upper limit kind using mask
# kinds: y upper-limit, x upper-limit
def map_limits_x(upp_mask_x, kind='west'):
    # use norm to define regular point
    return [kind if upp else 'norm' for upp in upp_mask_x]


# third define a function to map upper limit kind given two masks
# kinds: y upper-limit, x upper-limit, and xy upper-limit
def map_limits_xy(upp_mask_x, upp_mask_y):
    assert upp_mask_x.size == upp_mask_y.size, 'Could not map using arrays of different size'
    upp_mask_xy = upp_mask_x & upp_mask_y

    limits = []
    for i in range(upp_mask_x.size):
        if upp_mask_xy[i]:
            limits.append('southwest')
        elif upp_mask_x[i]:
            limits.append('west')
        elif upp_mask_y[i]:
            limits.append('south')
        else:
            limits.append('norm')
    return limits


# now given x-param and y-param,
def uppers_func(data, x_param, y_param, x_param_err, y_param_err):
    # assuming all are given then
    # calculate masks for x and y upper limits separately
    x_upp_mask = upp_mask(data, x_param, x_param_err)
    y_upp_mask = upp_mask(data, y_param, y_param_err)

    # now map masks to appropriate kind of arrow: west, southwest, and south
    limits = map_limits_xy(x_upp_mask, y_upp_mask)
    return limits


# the last function will only work if both parameters have errors
# let's not rely on that so,

def uppers_funcv2(data, x_param, y_param, x_param_err=None, y_param_err=None):
    # This function will take parameters and errors; the errors may be None
    if x_param_err and y_param_err:
        return uppers_func(data, x_param, y_param, x_param_err, y_param_err)
    elif x_param_err:
        z_upp_mask = upp_mask(data, x_param, x_param_err)
        return map_limits_x(z_upp_mask, kind='west')
    elif y_param_err:
        z_upp_mask = upp_mask(data, y_param, y_param_err)
        return map_limits_x(z_upp_mask, kind='south')
    else:
        raise ValueError('These parameters don\'t have upper limits!')


# I think the functions above will work for most purposes
# however, I need the original upper limit mask to make an array with the
# correct data coordinates to plot on; the ones with the placeholders replaced with the errors (upper limits)
# So instead of using the above we'll make a function that will return
# both the limits and the corrected parameter arrays

def correct_param(data, z_param, z_param_err, z_upp_mask):
    # Last modification: multiply upper limits by 2
    z_correct_params = data[z_param].copy()
    z_correct_params[z_upp_mask] = 2 * data[z_param_err][z_upp_mask]
    return z_correct_params


def limit_arrays(data, x_param, y_param, x_param_err=None, y_param_err=None):
    if x_param_err and y_param_err:
        x_upp_mask = upp_mask(data, x_param, x_param_err)
        y_upp_mask = upp_mask(data, y_param, y_param_err)

        x_correct_params = correct_param(data, x_param, x_param_err, x_upp_mask)
        y_correct_params = correct_param(data, y_param, y_param_err, y_upp_mask)
        limits = map_limits_xy(x_upp_mask, y_upp_mask)
    elif x_param_err:
        x_upp_mask = upp_mask(data, x_param, x_param_err)
        x_correct_params = correct_param(data, x_param, x_param_err, x_upp_mask)
        y_correct_params = data[y_param]
        limits = map_limits_x(x_upp_mask, kind='west')
    elif y_param_err:
        y_upp_mask = upp_mask(data, y_param, y_param_err)
        y_correct_params = correct_param(data, y_param, y_param_err, y_upp_mask)
        x_correct_params = data[x_param]
        limits = map_limits_x(y_upp_mask, kind='south')
    else:
        x_correct_params = data[x_param]
        y_correct_params = data[y_param]
        limits = None
        # raise ValueError('These parameters don\'t have upper limits!')
    return x_correct_params, y_correct_params, limits


""" 
TODO: So... drawing double upper limits using diagonals was kinda discouraged so I may want to make a function 
that will return the data split between normal points + upper limits, and double upper limits.
and then draw the upper limits twice; one time with left arrow and then the down arrow. 
"""

