import torch
"""
THE MAIN GOAL OF CONSTAINTS IS TO ELIMINATE ARBITAGE MEANING FREE, RISK FREE MONEY
"""
"""
Calender Spread Penalty:
This is a penalty that is used to enforce the rule that longer dated options must always have higher 
or equal IV than shorter dated ones. IV has to increase as time increases as the further something
is, the more uncertainty there is, the more the IV increases.
"""
def calendar_spread_penalty(model, moneyness_vals, time_vals, weight=1.0):
    """
    Penalize the model if IV decreases as time increases.
    Rule: longer dated options must always have higher or equal IV than shorter dated ones.
    We enforce this by checking dIV/dT > 0 everywhere on the surface.
    """
    T = torch.tensor(time_vals, dtype=torch.float32, requires_grad=True)
    K = torch.tensor(moneyness_vals, dtype=torch.float32)

    X = torch.stack([K, torch.log(T)], dim=1)
    iv = model(X)

    grad_T = torch.autograd.grad( #this basically takes the derivative of the IV with respect to time
        outputs=iv.sum(),
        inputs=T,
        create_graph=True
    )[0]

    penalty = torch.relu(-grad_T).mean() #if the derivative is negative, then the penalty is positive
    return weight * penalty

"""
Butterfly Spread Penalty:
This is a penalty that is used to enforce the rule that the IV must curve upward, never downward.
The smile must stay a smile because if it dips down that means that there is negative IV.
"""

def butterfly_spread_penalty(model, moneyness_vals, time_vals, weight=1.0):
    """
    Penalize the model if the vol smile is concave (not convex).
    Rule: the smile must curve upward, never downward.
    We enforce this by checking d2IV/dK2 > 0 everywhere.
    """
    K = torch.tensor(moneyness_vals, dtype=torch.float32, requires_grad=True)
    T = torch.tensor(time_vals, dtype=torch.float32)

    X = torch.stack([K, torch.log(T)], dim=1)
    iv = model(X)

    grad_K = torch.autograd.grad(
        outputs=iv.sum(),
        inputs=K,
        create_graph=True
    )[0]

    grad_KK = torch.autograd.grad(
        outputs=grad_K.sum(),
        inputs=K,
        create_graph=True
    )[0]

    penalty = torch.relu(-grad_KK).mean()
    return weight * penalty


def total_arbitrage_penalty(model, n_sample=64, weight=0.5):
    """
    Instead of checking every point on the surface for every combination of moneyness and time to expiry,
    we just sample 64 points and check those.
    Then you run both the penalties on those 64 points and then add them together.
    """
    moneyness = torch.FloatTensor(n_sample).uniform_(0.7, 1.3)
    time = torch.FloatTensor(n_sample).uniform_(0.05, 1.0)

    cal_pen = calendar_spread_penalty(model, moneyness, time, weight)
    but_pen = butterfly_spread_penalty(model, moneyness, time, weight)

    return cal_pen + but_pen