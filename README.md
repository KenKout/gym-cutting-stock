# Gynasium Cutting Stock Environment

## How to install
    
```bash
pip install git+https://github.com/martinakaduc/gym-cutting-stock
```

## Logging Configuration

You can control the verbosity of logging output using `set_log_level`:

```python
from gym_cutting_stock.logging_config import set_log_level
import logging

# Set logging level - options are:
set_log_level(logging.DEBUG)    # Most verbose
set_log_level(logging.INFO)     # Default level
set_log_level(logging.WARNING)  # Only warnings and errors
set_log_level(logging.ERROR)    # Only errors

# You can also use string names:
set_log_level('DEBUG')
set_log_level('INFO')
```

## Sample code for Random Policy

```python
import gym_cutting_stock
import random
import gymnasium as gym
env = gym.make("gym_cutting_stock/CuttingStock-v0", render_mode="human")


def random_policy(observation, info):
    list_prods = observation["products"]

    prod_size = [0, 0]
    stock_idx = -1
    pos_x, pos_y = 0, 0

    # Pick a product that has quality > 0
    for prod in list_prods:
        if prod["quantity"] > 0:
            prod_size = prod["size"]

            # Random choice a stock idx
            pos_x, pos_y = None, None
            for _ in range(100):
                # Random choice a stock
                stock_idx = random.randint(0, len(observation["stocks"]) - 1)
                stock = observation["stocks"][stock_idx]

                # Random choice a position
                stock_w, stock_h = stock.shape
                prod_w, prod_h = prod_size

                if stock_w < prod_w or stock_h < prod_h:
                    continue

                pos_x = random.randint(0, stock_w - prod_w)
                pos_y = random.randint(0, stock_h - prod_h)
                break

            if pos_x is not None and pos_y is not None:
                break

    return {
        "stock_idx": stock_idx,
        "size": prod_size,
        "position": (pos_x, pos_y)
    }


if __name__ == "__main__":
    observation, info = env.reset(seed=42)
    for _ in range(200):
        action = random_policy(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

env.close()
```