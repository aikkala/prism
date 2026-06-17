from typing import Any
import numpy as np

def describe_minigrid_map(env_state: Any) -> str:
    """
    Generates a text description of the MiniGrid map from the environment state.
    Assumes env_state is the unwrapped MiniGrid environment object.
    """
    grid = env_state.grid
    width = grid.width
    height = grid.height
    
    # Simple ASCII-like or text description
    desc = f"Map Size: {width}x{height}\n"
    
    objects = []
    for i in range(width):
        for j in range(height):
            cell = grid.get(i, j)
            if cell is not None and cell.type != 'wall':
                objects.append(f"- {cell.type} at ({i}, {j})")
    
    if objects:
        desc += "Objects on the map:\n" + "\n".join(objects) + "\n"
    else:
        desc += "The map is empty except for walls.\n"
        
    return desc
