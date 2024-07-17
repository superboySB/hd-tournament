class PIDController:
    def __init__(self, kp=0.05, ki=0.01, kd=0.02):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = {}
        self.previous_error = {}

    def update(self, error, key, control_type):
        if key not in self.integral:
            self.integral[key] = {}
        if key not in self.previous_error:
            self.previous_error[key] = {}

        self.integral[key][control_type] = self.integral[key].get(control_type, 0) + error
        derivative = error - self.previous_error[key].get(control_type, 0)
        self.previous_error[key][control_type] = error

        return self.kp * error + self.ki * self.integral[key][control_type] + self.kd * derivative

    def reset(self, key):
        if key in self.integral:
            self.integral[key] = {}
        if key in self.previous_error:
            self.previous_error[key] = {}