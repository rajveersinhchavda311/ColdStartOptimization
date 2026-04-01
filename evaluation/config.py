"""
Configuration: Cost parameters for autoscaling evaluation.

These parameters define the trade-off between cold starts and idle provisioning:
    COST_COLD: Penalty per unserved request (cold start)
    COST_IDLE: Cost per idle container-second

Ratio (10:1 default) reflects industry practice: service failures are orders of
magnitude more costly than over-provisioning.
"""

# Cost per unserved request (cold start penalty)
# Higher value = more aggressive pre-warming (fewer cold starts, more idle)
COST_COLD = 10.0

# Cost per idle container-second
# Normalized to 1.0 (infrastructure baseline cost)
COST_IDLE = 1.0
