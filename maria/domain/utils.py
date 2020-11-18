

def beta_parameters(shape, scale):
    alpha = shape*scale/(1-shape)
    beta = shape - 1 + shape*pow((1-shape),2)/pow(scale,2)
    return alpha, beta