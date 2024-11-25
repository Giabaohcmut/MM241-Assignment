from policy import Policy
import numpy as np
from numpy import random
import copy
from sortedcontainers import SortedList
from math import exp


def Policy2310790_2313873_2311011_2311770_2310271(Policy):
    def __init__(self):
        self.reset()

    def reset(self):
        self.first_time = True
        self.step = 0
        self.num_stocks = 0
        self.num_products = 0
        self.num_items = 0
        self.stocks = []
        self.stock_indices = None
        self.products = []
        self.product_indices = None
        self.state_matrix = []
        self.actions = []

    def get_action(self, observation, info):
        pass
    def greedy(self):
        # Sort the stocks array in descending order of stock areas.
        self.stock_indices, _ = zip(
            *sorted(zip(self.stock_indices, self.stocks), key = lambda x: -x[1]["width"] * x[1]["height"])
        )

        # Sort the products array in descending order of item areas.
        self.product_indices, _ = zip(
            *sorted(zip(self.product_indices, self.products), key = lambda x: -x[1]["width"] * x[1]["height"])
        )

        # Place the items into stocks according to greedy algorithm.
        for product_index in self.product_indices:
            starting_stock_index = 0
            product_width = self.products[product_index]["width"]
            product_height = self.products[product_index]["height"]
            product_demands = self.products[product_index]["demands"]

            for _ in range(product_demands):
                # Find the first largest stock where the item can be placed.
                for stock_index in range(starting_stock_index, self.num_stocks):
                    stock_index = self.stock_indices[stock_index]

                    # Find the appropriate position in the stock to place the item.
                    # If a appropriate position is found, move to the next item.
                    if self.place_item(self.stocks[stock_index], product_width, product_height):
                        break

                    starting_stock_index += 1
        
        # Trying to find smaller stocks to place items
        self.tightening()
        # Insert items to state matrix
        for i, stock in enumerate(self.stocks):
            for x, y, w, h in stock["products"]:
                self.state_matrix.append([w, h, i, x, y])
        # Convert the state matrix to a NumPy array for better performance during copying
        self.state_matrix = np.array(self.state_matrix, dtype = np.int32)

    # Find the appropriate position in the stock to place the item,
    # if no such position is found, return None.
    def place_item(self, stock, product_width, product_height):
        occupied_cells = stock["occupied_cells"]
        num_row = len(occupied_cells)
        num_col = len(occupied_cells[0])
        verticals, horizontals = stock["grid"]

        # Loop through each cell of the grid and find the cluster of cells that are not occupied,
        # where the item can be placed.
        for i in range(num_row):
            for j in range(num_col):
                if not occupied_cells[i][j]:
                    # Check if there is enough space to the right of the current cell.
                    right_edge = None
                    for k in range(j + 1, num_col + 1):
                        if occupied_cells[i][k - 1]:
                            break

                        if verticals[k] >= verticals[j] + product_width:
                            right_edge = k
                            break

                    # Check if there is enough space above the current cell.
                    if right_edge != None:
                        for k in range(i + 1, num_row + 1):
                            if occupied_cells[k - 1][j]:
                                break

                            if horizontals[k] >= horizontals[i] + product_height:
                                # Update the grid after the item is placed.
                                if verticals[j] + product_width < verticals[right_edge]:
                                    verticals.add(verticals[j] + product_width)
                                    for row in occupied_cells:
                                        row.insert(right_edge, row[right_edge - 1])

                                if horizontals[i] + product_height < horizontals[k]:
                                    horizontals.add(horizontals[i] + product_height)
                                    occupied_cells.insert(k, copy.deepcopy(occupied_cells[k - 1]))

                                # Update the occupied cells after the item is placed.
                                for m in range(i, k):
                                    for n in range(j, right_edge):
                                        occupied_cells[m][n] = True
                                
                                # Place the item
                                stock["products"].append(
                                    [verticals[j], horizontals[i], product_width, product_height]
                                )
                                # Update inner bound
                                if stock["right_bound"] < verticals[j] + product_width:
                                    stock["right_bound"] = verticals[j] + product_width
                                if stock["top_bound"] < horizontals[i] + product_height:
                                    stock["top_bound"] = horizontals[i] + product_height
                                return True
        
        return False
    
    def tightening(self):
        # Sort the stocks array in descending order of wasted areas.
        wasted_indices = np.arange(self.num_stocks)
        wasted_indices, _ = zip(
            *sorted(
                zip(wasted_indices, self.stocks), 
                key = lambda x: float("inf") if x[1]["right_bound"] * x[1]["top_bound"] == 0 
                else x[1]["right_bound"] * x[1]["top_bound"] - x[1]["width"] * x[1]["height"]
            )
        )

        for stock_index in wasted_indices:
            stock = self.stocks[stock_index]
            for i in self.stock_indices[::-1]:
                replace_stock = self.stocks[i]
                if replace_stock["top_bound"] * replace_stock["right_bound"] == 0 and \
                   replace_stock["width"] >= stock["right_bound"] and replace_stock["height"] >= stock["top_bound"] and \
                   replace_stock["width"] * replace_stock["height"] < stock["width"] * stock["height"]:
                    replace_stock["products"] = stock["products"]
                    replace_stock["right_bound"] = stock["right_bound"]
                    replace_stock["top_bound"] = stock["top_bound"]

                    stock["products"] = []
                    stock["right_bound"] = 0
                    stock["top_bound"] = 0
                    break
                       
    # Calculate the probability of transitioning to the next state.
    def transition_probability(self, current_state_energy, next_state_energy, temperature):
        if next_state_energy <= current_state_energy:
            return 1
        
        return exp((current_state_energy - next_state_energy) / temperature)
    

    # Check if the position for placing the item is within the stock's bounds.
    def is_inside(self, stock_width, stock_height, product_size, position):
        return product_size[0] + position[0] <= stock_width and product_size[1] + position[1] <= stock_height
    
    # Helper function to generate a random integer centered around 0, following a normal distribution.
    def rand(self, std):
        return int(random.randn() * std)

    # Choose a random next state that is near the current state.
    def choose_next_state(self, current_state, overlap_register):
        next_state = current_state.copy()
        prob_list = np.array(overlap_register, dtype = np.float32)
        overlap_num = prob_list.sum()
        if overlap_num > 0:
            prob_list += 0.3 * overlap_num / (0.7 * self.num_items - overlap_num)
            prob_list /= prob_list.sum()
        else:
            prob_list += 1
            prob_list /= self.num_items

        # Swap the positions of two randomly selected items that are near each other.
        if random.randint(0, 100) > 50:
            i = random.randint(0, self.num_items)
            j = min(max(0, i + self.rand(20)), self.num_items - 1)
            wi, hi, zi, xi, yi = next_state[i]
            wj, hj, zj, xj, yj = next_state[j]

            if self.is_inside(self.stocks[zi]["width"], self.stocks[zi]["height"], (wj, hj), (xi, yi)) and \
               self.is_inside(self.stocks[zj]["width"], self.stocks[zj]["height"], (wi, hi), (xj, yj)):
                next_state[i][:2] = [wj, hj]
                next_state[j][:2] = [wi, hi]
        
        # Move a randomly selected item to another randomly selected stock, prioritizing items with overlap.
        if random.randint(0, 100) > 40:
            i = random.choice(self.num_items, p = prob_list)
            wi, hi, zi, xi, yi = next_state[i]
            new_stock_index = min(max(0, zi + self.rand(15)), self.num_stocks - 1)

            if self.is_inside(self.stocks[new_stock_index]["width"], self.stocks[new_stock_index]["height"], (wi, hi), (xi, yi)):
                next_state[i][2] = new_stock_index

        # Move a randomly selected item to a nearby random location, prioritizing items with overlap.
        if random.randint(0, 100) > 10:
            i = random.choice(self.num_items, p = prob_list)
            wi, hi, zi, xi, yi = next_state[i]

            next_state[i][3:] = [
                min(max(0, xi + self.rand(30)), self.stocks[zi]["width"] - wi),
                min(max(0, yi + self.rand(30)), self.stocks[zi]["height"] - hi)
            ]

        return next_state
