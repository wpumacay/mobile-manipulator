import random

def generate_maze_walls(width, height):
    """
    Generate walls for a random maze using recursive backtracking algorithm.
    Returns a list of wall coordinates as tuples [(x_start, y_start), (x_end, y_end)].
    
    Args:
        width (int): Width of the maze in cells
        height (int): Height of the maze in cells
    
    Returns:
        list: List of wall coordinate tuples
    """
    # Initialize the grid
    grid = [[False] * width for _ in range(height)]
    walls = []
    
    def is_valid(x, y):
        return 0 <= x < width and 0 <= y < height
    
    def get_neighbors(x, y):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
            new_x, new_y = x + dx*2, y + dy*2
            if is_valid(new_x, new_y) and not grid[new_y][new_x]:
                neighbors.append((new_x, new_y, dx, dy))
        return neighbors
    
    def carve_path(x, y):
        grid[y][x] = True
        
        # Get all unvisited neighbors
        neighbors = get_neighbors(x, y)
        random.shuffle(neighbors)
        
        for next_x, next_y, dx, dy in neighbors:
            if not grid[next_y][next_x]:
                # Mark the cell between current and next as visited (carve the path)
                grid[y + dy][x + dx] = True
                carve_path(next_x, next_y)
    
    # Start from cell (0, 0)
    carve_path(0, 0)
    
    # Generate walls based on the grid
    # Vertical walls
    for x in range(width + 1):
        wall_start = None
        for y in range(height + 1):
            should_have_wall = (
                x == 0 or x == width or
                (x < width and y < height and
                 (x == 0 or not grid[y][x-1] or not grid[y][x]))
            )
            
            if should_have_wall and wall_start is None:
                wall_start = (x, y)
            elif not should_have_wall and wall_start is not None:
                walls.append((wall_start, (x, y)))
                wall_start = None
        if wall_start is not None:
            walls.append((wall_start, (x, height)))
    
    # Horizontal walls
    for y in range(height + 1):
        wall_start = None
        for x in range(width + 1):
            should_have_wall = (
                y == 0 or y == height or
                (y < height and x < width and
                 (y == 0 or not grid[y-1][x] or not grid[y][x]))
            )
            
            if should_have_wall and wall_start is None:
                wall_start = (x, y)
            elif not should_have_wall and wall_start is not None:
                walls.append((wall_start, (x, y)))
                wall_start = None
        if wall_start is not None:
            walls.append((wall_start, (width, y)))
    
    return walls

# Example usage
if __name__ == "__main__":
    # Generate a 5x5 maze
    maze_walls = generate_maze_walls(5, 5)
    
    # Print the walls
    for wall in maze_walls:
        print(f"Wall from {wall[0]} to {wall[1]}")