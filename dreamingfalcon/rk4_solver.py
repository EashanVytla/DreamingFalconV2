class RK4_Solver:
    def __init__(self, dt):
        self.dt = dt
    
    def step(self, x, f, *args):
        """
        Performs one step of RK4 integration
        
        Args:
            x: Current state
            f: Function that computes derivatives (dx/dt = f(x, t, *args))
            *args: Additional arguments to pass to f
            
        Returns:
            Next state after dt
        """
        h = self.dt
        
        # RK4 steps
        k1 = f(x, *args)
        k2 = f(x + h/2 * k1, *args)
        k3 = f(x + h/2 * k2, *args)
        k4 = f(x + h * k3, *args)
        
        # Update state
        return x + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)