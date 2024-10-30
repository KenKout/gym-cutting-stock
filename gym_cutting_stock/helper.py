"""
Metrics calculation utilities for the cutting stock environment.
"""

import numpy as np
from gym_cutting_stock.logging_config import logger


class MetricsCalculator:
    """Calculates efficiency metrics for the cutting stock problem."""
    def __init__(self, observation, initial_products, initial_stocks):
        """
        Initialize with the current observation, initial products, and initial stocks from the environment.
        """
        self.stocks = observation['stocks']
        self.current_products = observation['products']
        self.initial_products = initial_products
        self.initial_stocks = initial_stocks

    def _get_usable_stock_area(self):
        """
        Calculate the total usable stock area based on the initial stock states.
        """
        total_stock_area = 0
        for i, stock in enumerate(self.initial_stocks):
            usable_cells = np.count_nonzero(stock == -1)
            total_stock_area += usable_cells
            logger.debug(f"Stock {i}: Usable area={usable_cells}")
        logger.debug(f"Total usable stock area: {total_stock_area}")
        return total_stock_area

    def _get_placed_area(self):
        """
        Calculate the total area of newly placed products by comparing initial and current stock states.
        Assumes that:
            - `-1` indicates initially unused cells.
            - Any value != -1 and == -2 indicates occupied by a product.
        Returns:
            - total_product_placement_area: Total area occupied by newly placed products
            - num_used_stocks: Number of stock sheets containing placed products
            - total_available_area: Total usable area across all used stock sheets
        """
        total_product_placement_area = 0  # Total area occupied by newly placed products
        num_used_stocks = 0  # Number of stock sheets containing placed products
        total_available_area = 0  # Total usable area in used stock sheets

        for stock_index in range(len(self.stocks)):
            initial_stock = self.initial_stocks[stock_index]
            current_stock = self.stocks[stock_index]

            # Find cells that were empty (-1) initially and are now occupied by products
            newly_placed_cells = np.logical_and(initial_stock == -1, current_stock != -1)
            area_of_new_placements = np.count_nonzero(newly_placed_cells)
            total_product_placement_area += area_of_new_placements

            if area_of_new_placements > 0:
                logger.debug(f"Stock {stock_index}: Newly placed cells={area_of_new_placements}")
                num_used_stocks += 1  # Count stocks with newly placed products

                # Calculate total available area in this stock sheet
                total_available_area += np.count_nonzero(initial_stock == -1)
            else:
                logger.debug(f"Stock {stock_index}: No new placements")

        logger.debug(f"Total area occupied by products: {total_product_placement_area}")
        return total_product_placement_area, num_used_stocks, total_available_area

    def _calculate_total_product_area(self):
        """
        Calculate the total area of placed products by comparing initial and remaining quantities.
        Returns the total area of all products that have been successfully placed.
        """
        total_initial_product_area = 0  # Total area of all products at start
        total_unplaced_product_area = 0  # Total area of products not yet placed
        for initial_product in self.initial_products:
            size = initial_product['size']
            initial_quantity = initial_product['quantity']

            # Find the corresponding current product by size
            current_product = next(
                (p for p in self.current_products if np.array_equal(p['size'], initial_product['size'])),
                None
            )
            if current_product is None:
                # If the product was fully placed (quantity reduced to 0)
                remaining_quantity = 0
                logger.debug(f"Product {size}: Fully placed (remaining_quantity=0)")
            else:
                remaining_quantity = current_product['quantity']
                logger.debug(f"Product {size}: Remaining quantity={remaining_quantity}")

            placed_quantity = initial_quantity - remaining_quantity
            placed_area = size[0] * size[1] * placed_quantity
            total_initial_product_area += size[0] * size[1] * initial_quantity
            total_unplaced_product_area += size[0] * size[1] * remaining_quantity

            logger.debug(
                f"Product {size}: Initial area={size[0] * size[1] * initial_quantity}, Remaining area={size[0] * size[1] * remaining_quantity}, Placed area={placed_area}")

        total_placed_product_area = total_initial_product_area - total_unplaced_product_area
        logger.debug(f"Total initial product area: {total_initial_product_area}")
        logger.debug(f"Total area of unplaced products: {total_unplaced_product_area}")
        logger.debug(f"Total area of placed products: {total_placed_product_area}")
        return total_placed_product_area

    def get_filled_ratio(self):
        """
        Calculate the filled ratio as the total placed product area divided by the total used stock area.
        """
        total_placed_product_area = self._calculate_total_product_area()
        total_usable_stock_area = self._get_usable_stock_area()
        total_placed_area, used_stocks, total_stock_area = self._get_placed_area()

        if total_stock_area == 0:
            logger.warning("Total stock area is zero. Filled ratio set to 0.")
            return 0.0

        # Debug logging
        logger.debug(f"Total placed product area: {total_usable_stock_area}")
        logger.debug(f"Total placed area: {total_placed_area} and total placed product area: {total_placed_product_area}. Is equal: {total_placed_area == total_placed_product_area}")
        logger.debug(f"Total used stocks: {used_stocks}")
        logger.debug(f"Total stock that contains newly placed products: {total_stock_area}")

        # Calculate filled ratio
        filled_ratio = total_placed_product_area / total_stock_area
        logger.debug(f"Filled ratio calculation: {total_placed_product_area} / {total_stock_area} = {filled_ratio}")
        return filled_ratio
