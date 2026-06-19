"""
MOPSO-Based CNN for Keyword Selection on Google Ads
mopso_optimizer.py — Multi-Objective Particle Swarm Optimisation

Optimises CNN hyperparameters to simultaneously:
  - Maximise classification accuracy
  - Minimise training time

B.E. Final Year Project, AIT Chikkamagaluru (VTU), 2020-21
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Particle:
    """Represents one candidate CNN configuration in the swarm."""
    position: np.ndarray        # CNN hyperparameters
    velocity: np.ndarray        # Direction of change
    best_position: np.ndarray   # Personal best found so far
    best_objectives: Tuple      # (accuracy, -training_time) at best position
    objectives: Tuple = (0.0, 0.0)  # Current objectives


class MOPSO:
    """
    Multi-Objective Particle Swarm Optimisation for CNN architecture search.

    Search space — 5 hyperparameters:
        [0] num_filters     : number of conv filters  (32–256)
        [1] kernel_size     : conv kernel size         (2–5)
        [2] dropout_rate    : dropout probability      (0.1–0.7)
        [3] learning_rate   : Adam LR (scaled x1000)  (0.1–10 → 0.0001–0.01)
        [4] dense_units     : units in dense layer     (64–512)

    Objectives (maximise both):
        obj_1 = validation accuracy
        obj_2 = 1 / training_time  (faster training = higher score)
    """

    # Hyperparameter bounds [min, max]
    BOUNDS = np.array([
        [32,  256],     # num_filters
        [2,   5],       # kernel_size
        [0.1, 0.7],     # dropout_rate
        [0.1, 10.0],    # learning_rate (x1000)
        [64,  512],     # dense_units
    ])

    def __init__(
        self,
        n_particles: int = 20,
        n_iterations: int = 30,
        w: float = 0.7,     # inertia weight
        c1: float = 1.5,    # cognitive coefficient
        c2: float = 1.5,    # social coefficient
        seed: int = 42
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        np.random.seed(seed)

        self.swarm: List[Particle] = []
        self.pareto_front: List[Particle] = []
        self.history = []

    def _random_position(self) -> np.ndarray:
        """Initialise a random position within bounds."""
        return np.array([
            np.random.uniform(low, high)
            for low, high in self.BOUNDS
        ])

    def _decode(self, position: np.ndarray) -> dict:
        """Decode continuous position vector into CNN hyperparameters."""
        return {
            'num_filters':   int(np.clip(position[0], 32, 256)),
            'kernel_size':   int(np.clip(position[1], 2, 5)),
            'dropout_rate':  float(np.clip(position[2], 0.1, 0.7)),
            'learning_rate': float(np.clip(position[3], 0.1, 10.0)) / 1000,
            'dense_units':   int(np.clip(position[4], 64, 512)),
        }

    def _evaluate(self, position: np.ndarray, train_fn) -> Tuple[float, float]:
        """
        Evaluate a particle by training a CNN with its hyperparameters.

        Args:
            position: Particle position vector
            train_fn: Function that accepts hyperparameters dict and returns
                      (val_accuracy, training_time_seconds)

        Returns:
            (accuracy_obj, time_obj) — both to maximise
        """
        params = self._decode(position)
        start = time.time()
        val_accuracy, training_time = train_fn(params)
        elapsed = time.time() - start

        accuracy_obj = val_accuracy
        time_obj = 1.0 / max(elapsed, 0.001)   # faster = higher score

        return accuracy_obj, time_obj

    def _dominates(self, obj_a: Tuple, obj_b: Tuple) -> bool:
        """Return True if obj_a Pareto-dominates obj_b (higher is better for both)."""
        return (
            all(a >= b for a, b in zip(obj_a, obj_b)) and
            any(a > b for a, b in zip(obj_a, obj_b))
        )

    def _update_pareto_front(self):
        """Rebuild Pareto front from current swarm."""
        candidates = self.swarm[:]
        pareto = []
        for p in candidates:
            dominated = False
            for q in candidates:
                if q is not p and self._dominates(q.best_objectives, p.best_objectives):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)
        self.pareto_front = pareto

    def _select_global_best(self) -> np.ndarray:
        """Select a random member of the Pareto front as global guide."""
        if not self.pareto_front:
            return self.swarm[0].best_position
        return np.random.choice(self.pareto_front).best_position

    def initialise(self, train_fn):
        """Initialise swarm with random positions and evaluate."""
        print(f"Initialising swarm ({self.n_particles} particles)...")
        self.swarm = []

        for i in range(self.n_particles):
            pos = self._random_position()
            vel = np.zeros_like(pos)
            objs = self._evaluate(pos, train_fn)

            p = Particle(
                position=pos.copy(),
                velocity=vel,
                best_position=pos.copy(),
                best_objectives=objs,
                objectives=objs
            )
            self.swarm.append(p)
            print(f"  Particle {i+1:2d}: accuracy={objs[0]:.4f}, params={self._decode(pos)}")

        self._update_pareto_front()

    def optimise(self, train_fn) -> dict:
        """
        Run MOPSO optimisation loop.

        Args:
            train_fn: Callable(params: dict) → (val_accuracy: float, training_time: float)

        Returns:
            Best hyperparameters found (by accuracy on Pareto front)
        """
        self.initialise(train_fn)

        for iteration in range(self.n_iterations):
            print(f"\nIteration {iteration + 1}/{self.n_iterations}")
            gbest = self._select_global_best()

            for p in self.swarm:
                r1 = np.random.rand(len(p.position))
                r2 = np.random.rand(len(p.position))

                # Standard PSO velocity update
                p.velocity = (
                    self.w * p.velocity +
                    self.c1 * r1 * (p.best_position - p.position) +
                    self.c2 * r2 * (gbest - p.position)
                )

                p.position = np.clip(
                    p.position + p.velocity,
                    self.BOUNDS[:, 0],
                    self.BOUNDS[:, 1]
                )

                p.objectives = self._evaluate(p.position, train_fn)

                # Update personal best if current position is better on either objective
                if (p.objectives[0] > p.best_objectives[0] or
                        p.objectives[1] > p.best_objectives[1]):
                    p.best_position = p.position.copy()
                    p.best_objectives = p.objectives

            self._update_pareto_front()

            best_acc = max(p.best_objectives[0] for p in self.pareto_front)
            print(f"  Pareto front size: {len(self.pareto_front)}, best accuracy: {best_acc:.4f}")
            self.history.append({'iteration': iteration + 1, 'best_accuracy': best_acc})

        return self.get_best_params()

    def get_best_params(self) -> dict:
        """Return the Pareto-front solution with highest accuracy."""
        best = max(self.pareto_front, key=lambda p: p.best_objectives[0])
        params = self._decode(best.best_position)
        print(f"\nBest parameters found:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print(f"  Validation accuracy: {best.best_objectives[0]:.4f}")
        return params


if __name__ == '__main__':
    # Demo with a mock training function
    def mock_train(params: dict):
        """Simulate CNN training — replace with real training in practice."""
        time.sleep(0.1)
        # Simulate accuracy increasing with more filters and lower dropout
        acc = (
            0.5 +
            (params['num_filters'] / 512) * 0.2 +
            (1 - params['dropout_rate']) * 0.15 +
            np.random.uniform(-0.05, 0.05)
        )
        return min(acc, 0.9386), 0.1

    mopso = MOPSO(n_particles=5, n_iterations=3)
    best = mopso.optimise(mock_train)
    print("\nOptimisation complete.")
    print(f"Best config: {best}")
