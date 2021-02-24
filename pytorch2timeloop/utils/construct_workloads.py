import functools
import yaml
import pkgutil

def prod (l):
    return functools.reduce(lambda x, y: x*y, l)


def rewrite_workload_bounds(dst, workload_bounds):
    # print(workload_bounds)
    # mode, w, h, c, n, m, s, r, wpad, hpad, wstride, hstride, g, b = workload_bounds
    mode, w, h, c, n, m, s, r, wpad, hpad, wstride, hstride, g = workload_bounds
    q = int((w - s + 2 * wpad) / wstride) + 1
    p = int((h - r + 2 * hpad) / hstride) + 1


    if mode == "norm-conv" or mode == 'linear':
        
        # Modify (Feb 1, 2021) - Kyungmi
        # Use the util function instead of using the relative path file loading
        # with open(src, "r") as f:
        #     config = yaml.load(f, Loader = yaml.SafeLoader)
        config = get_convolution_workload()

        config['problem']['instance']['R'] = r
        config['problem']['instance']['S'] = s
        config['problem']['instance']['P'] = p
        config['problem']['instance']['Q'] = q
        config['problem']['instance']['C'] = c
        config['problem']['instance']['M'] = m
        config['problem']['instance']['N'] = n
        config['problem']['instance']['Wstride'] = wstride
        config['problem']['instance']['Hstride'] = hstride

        with open(dst, "w") as f:
            f.write(yaml.dump(config))

    
    elif mode == "depth-wise":

        # Similar as the above modification
        # with open(src, "r") as f:
        #     config = yaml.load(f, Loader = yaml.SafeLoader)
        config = get_depthwise_workload()

        config['problem']['instance']['R'] = r
        config['problem']['instance']['S'] = s
        config['problem']['instance']['P'] = p
        config['problem']['instance']['Q'] = q
        config['problem']['instance']['C'] = c
        # config['problem']['instance']['M'] = m
        config['problem']['instance']['N'] = n
        config['problem']['instance']['Wstride'] = wstride
        config['problem']['instance']['Hstride'] = hstride

        with open(dst, "w") as f:
            f.write(yaml.dump(config))
    

    else:
        
        print("Error: DNN Layer Type Not Supported")
        return

    
    # print("scaffold file --> {}".format(src))
    print("workload file --> {}".format(dst))
    

def get_convolution_workload():
    f = pkgutil.get_data("pytorch2timeloop", "utils/convolution.yaml")
    config_conv = yaml.load(f, Loader = yaml.SafeLoader)
    return config_conv

def get_depthwise_workload():
    f = pkgutil.get_data("pytorch2timeloop", "utils/depth_wise_convolution.yaml")
    config_depth = yaml.load(f, Loader = yaml.SafeLoader)
    return config_depth
