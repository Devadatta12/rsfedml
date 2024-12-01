"""fedavg: A Flower / TensorFlow app."""


from flwr.common import Context, ndarrays_to_parameters, FitRes, Scalar, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy, EvaluateRes
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Union, Optional, Dict
from logging import WARNING
from typing import Callable, Optional, Union
import csv
from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from functools import partial, reduce
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from fedrr.task import load_model
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy
import pandas as pd
import random


def round_robin(server_round: int, results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
    """Compute in-place weighted average."""
    round_robin_index = server_round%len(results)
    print(f"Selected client {results[round_robin_index][0].cid} for round {server_round}")
    params = [
        x for x in parameters_to_ndarrays(results[round_robin_index][1].parameters)
    ]

    return params
    

class AggregateCustomMetricStrategy(FedAvg):


    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]],
                      failures: list[Union[tuple[ClientProxy, FitRes], BaseException]]) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        print(f"Number of clients is {len(results)} for round {server_round}")
        
        aggregated_ndarrays = round_robin(server_round, results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        max_accuracy = max(accuracies)
        print(
            f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}"
        )

        with open("metrics.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if server_round == 1:
                writer.writerow(["Round", "Num Clients", "Loss", "Accuracy", "Max Accuracy"])

            writer.writerow([
                server_round,
                len(results),
                aggregated_loss,
                aggregated_accuracy,
                max_accuracy
            ])

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}
    


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = strategy = AggregateCustomMetricStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
