"""Geometry utility functions."""
from typing import Tuple, List
import math
import random

Point = Tuple[float, float]

def calculate_hex_vertices(center: Point, radius: float) -> List[Point]:
    """Calculate vertices for a flat-top hexagon.
    
    Args:
        center: The center point (x, y) of the hexagon
        radius: The radius of the hexagon
        
    Returns:
        List of vertex points in clockwise order
    """
    return [(center[0] + radius * math.cos(a), 
             center[1] - radius * math.sin(a))
            for a in [math.radians(angle) for angle in [120, 60, 0, 300, 240, 180]]]

def calculate_edge_points(vertices: List[Point]) -> List[Point]:
    """Calculate the midpoints of each edge.
    
    Args:
        vertices: List of vertex points
        
    Returns:
        List of edge midpoints
    """
    return [((vertices[i][0] + vertices[(i + 1) % 6][0]) / 2,
             (vertices[i][1] + vertices[(i + 1) % 6][1]) / 2)
            for i in range(6)]

def line_segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Check if line segments (p1,p2) and (p3,p4) intersect.
    
    Args:
        p1, p2: Points defining first line segment
        p3, p4: Points defining second line segment
        
    Returns:
        True if the line segments intersect
    """
    def ccw(A: Point, B: Point, C: Point) -> bool:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def point_in_hex(point: Point, center: Point, radius: float) -> bool:
    """Check if a point lies inside a regular hexagon.
    
    Args:
        point: The point to check
        center: The center of the hexagon
        radius: The radius of the hexagon
        
    Returns:
        True if the point is inside the hexagon
    """
    dx = abs(point[0] - center[0]) / radius
    dy = abs(point[1] - center[1]) / radius
    return (dx <= 1.0 and 
            dy <= math.sqrt(3)/2 and 
            dy <= math.sqrt(3) * (1 - dx/2))

def generate_jagged_line(start: Point, end: Point, 
                        num_segments: int, 
                        max_deviation: float) -> List[Point]:
    """Generate a jagged line between two points.
    
    Args:
        start: Starting point
        end: Ending point
        num_segments: Number of line segments
        max_deviation: Maximum deviation from straight line
        
    Returns:
        List of points defining the jagged line
    """
    points = [start]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    total_dist = math.sqrt(dx*dx + dy*dy)
    
    # Calculate perpendicular direction
    perp_dx = -dy / total_dist
    perp_dy = dx / total_dist
    
    for i in range(1, num_segments):
        t = i / num_segments
        base_x = start[0] + dx * t
        base_y = start[1] + dy * t
        
        # Deviation that reduces near endpoints
        deviation = random.uniform(-max_deviation, max_deviation) * total_dist
        deviation *= math.sin(t * math.pi)  # Reduce deviation at endpoints
        
        point = (base_x + perp_dx * deviation,
                base_y + perp_dy * deviation)
        points.append(point)
    
    points.append(end)
    return points 