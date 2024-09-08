"""
This script simulates a single-server M/M/1 queue and measures the mean response time
for different values of lambda (arrival rate). The job sizes are distributed according to
an exponential distribution with rate mu (service rate), while the interarrival times are
i.i.d. according to an exponential distribution with rate lambda.

Arguments:
--lambdas: List of lambda (arrival rate) values to test. Default is [0.5, 0.7, 0.9].
--mu: Service rate (mu) value. Default is 1.0.
--jobs: Number of jobs to simulate per run. Default is 2001.
--runs: Number of independent simulation runs. Default is 200.
--plot: Plot the generated RVs

Examples:
1. Run with default values:
   python hw_2_4_3_mm1_queue_simulation.py.py

2. Run with custom lambdas, mu, arrivals, and runs:
   python hw_2_4_3_mm1_queue_simulation.py.py --lambdas 0.6 0.8 1.0 --mu 0.9 --jobs 3000 --runs 100 --plot 0

Sample Response:

Mean response time for λ = 0.5: 0.1976
Total time taken for λ = 0.5: 1.54 seconds

Mean response time for λ = 0.7: 0.3157
Total time taken for λ = 0.7: 1.49 seconds

Mean response time for λ = 0.9: 0.9079
Total time taken for λ = 0.9: 1.50 seconds
"""
import argparse
import math
import random
import time

class MM1QueueSimulator:
    def __init__(self, lambd, mu):
        self.lambd = lambd
        self.mu = mu

    def exponential_random(self, rate):
        """Generate an exponential random variable using the inverse transform method."""
        return -math.log(1 - random.random()) / rate

    def exponential_random_np(self, rate):
        """Use numpy exponential method"""
        import numpy as np
        return np.random.exponential(scale=1/rate)

    def simulate(self, total_jobs):
        """Simulate the M/M/1 queue for the given number of jobs."""
        interarrival_times = [self.exponential_random(self.lambd) for _ in range(total_jobs + 1)]
        service_times = [self.exponential_random(self.mu) for _ in range(total_jobs + 1)]
        arrival_times = [sum(interarrival_times[:i+1]) for i in range(total_jobs + 1)]
        departure_times = [0] * (total_jobs + 1)

        for i in range(1, total_jobs + 1):
            # Calculate when the service can start: either after the previous job departs or as soon as it arrives, whichever is later.
            service_start_time = max(departure_times[i - 1], arrival_times[i])
            # Compute the departure time of the current job by adding the service time to the start time.
            departure_times[i] = service_start_time + service_times[i]

        response_time = departure_times[total_jobs] - arrival_times[total_jobs]

        return response_time, interarrival_times, service_times

class MM1QueueExperiment:
    def __init__(self, lambdas, mu, total_jobs, n_runs, plot_enabled=False):
        self.lambdas = lambdas
        self.mu = mu
        self.total_jobs = total_jobs
        self.n_runs = n_runs
        self.plot_enabled = plot_enabled

    def run(self):
        for lambd in self.lambdas:

            start_time = time.time()
            simulator = MM1QueueSimulator(lambd, self.mu)
            all_response_times = []
            all_interarrival_times = []
            all_service_times = []

            for _ in range(self.n_runs):
                response_time, interarrival_times, service_times = simulator.simulate(self.total_jobs)
                all_response_times.append(response_time)
                if self.plot_enabled:
                    all_interarrival_times.extend(interarrival_times)
                    all_service_times.extend(service_times)
            mean_response_time = sum(all_response_times)/self.total_jobs
            total_time = time.time() - start_time
            print(f"Mean response time for λ = {lambd}: {mean_response_time:.4f}")
            print(f"Total time taken for λ = {lambd}: {total_time:.2f} seconds\n")

            if self.plot_enabled:
                self.plot_histograms(lambd, all_response_times, all_interarrival_times, all_service_times)

    def plot_histograms(self, lambd, response_times, interarrival_times, service_times):
        """Plot histograms for response, interarrival, and service times."""
        import numpy as np
        import matplotlib.pyplot as plt

        plt.figure(figsize=(18, 6))

        # Interarrival Times
        plt.subplot(1, 3, 1)
        plt.hist(interarrival_times, bins=np.arange(0, max(interarrival_times) + 0.1, 0.1), alpha=0.7, color='blue')
        plt.title(f'Interarrival Times Histogram (λ={lambd})')
        plt.xlabel('Time')
        plt.ylabel('Frequency')

        # Service Times
        plt.subplot(1, 3, 2)
        plt.hist(service_times, bins=np.arange(0, max(service_times) + 0.1, 0.1), alpha=0.7, color='green')
        plt.title(f'Service Times Histogram (μ={self.mu})')
        plt.xlabel('Time')
        plt.ylabel('Frequency')

        # Response Times
        plt.subplot(1, 3, 3)
        plt.hist(response_times, bins=np.arange(0, max(response_times) + 0.1, 0.1), alpha=0.7, color='red')
        plt.title(f'Response Times Histogram (n={self.n_runs}, λ={lambd})')
        plt.xlabel('Time')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate an M/M/1 queue and plot results for multiple λ values.")
    parser.add_argument("--lambdas", type=float, nargs='+', default=[0.5, 0.7, 0.9], help="List of arrival rates (λ)")
    parser.add_argument("--mu", type=float, default=1, help="Rate of service (μ)")
    parser.add_argument("--jobs", type=int, default=2001, help="Total number of jobs to simulate")
    parser.add_argument("--runs", type=int, default=200, help="Number of simulation runs")
    parser.add_argument("--plot", type=int, default=0, help="Enable plotting of results")

    args = parser.parse_args()
    random.seed(1)
    experiment = MM1QueueExperiment(lambdas=args.lambdas, mu=args.mu, total_jobs=args.jobs, n_runs=args.runs, plot_enabled=bool(int(args.plot)))
    experiment.run()
