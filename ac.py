import model

class ConfidenceIntervals:
    def __init__(self, model_bounds):
        self.model_bounds = model_bounds

        self.sojourn_rate_intervals = []
        self.probability_estimates = []
        self.probability_errors = []

        raise Exception("finish")

    def get_abandonment_estimates(self):
        raise Exception("finish")

    def get_customer_arrival_rates(self):
        raise Exception("finish")

    def get_server_arrival_rates(self):
        raise Exception("finish")


class Exploration:
    def __init__(self, model_bounds):
        self.model_bounds = model_bounds
        raise Exception("finish")

def generate_model(confidence_intervals, state_rewards):
    pass
