{"username":"mahaboobandbasha","key":"f9a1a6c863f923aab4559f695aabeafb"}


def compute_centroid(corners):
    """Compute the centroid of a set of points."""
    x_sum, y_sum = 0, 0
    for x, y in corners:
        x_sum += x
        y_sum += y
    return x_sum / len(corners), y_sum / len(corners)

def compute_angle(centroid, corner):
    """Compute the angle of a corner with respect to the centroid."""
    dx = corner[0] - centroid[0]
    dy = corner[1] - centroid[1]
    return math.atan2(dy, dx)

def sort_corners(corners):
    """Sort corners based on their angle with respect to the centroid."""
    centroid = compute_centroid(corners)
    corners_with_angles = [(corner, compute_angle(centroid, corner)) for corner in corners]
    corners_with_angles.sort(key=lambda x: x[1])
    return [corner for corner, _ in corners_with_angles]

def verify_corner_order(corners):
    """Verify if the corners of a marker are in the right order."""
    # Assuming the right order is clockwise
    sorted_corners = sort_corners(corners)
    # Here you could add more checks, e.g., comparing to a known good order
    # For simplicity, we're just returning the sorted corners
    return sorted_corners

# Example usage:
import math

# Mock data: corners of a marker (x, y)
# Replace these with your actual corner points
corners = [(1, 2), (2, 3), (1, 3), (2, 2)]

sorted_corners = verify_corner_order(corners)
print("Sorted Corners:", sorted_corners)

