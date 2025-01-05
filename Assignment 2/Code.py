import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#######################################################################################################

def days_between_dates(date1, date2):
    dt1 = datetime(date1[0], date1[1], date1[2], date1[3], date1[4])
    dt2 = datetime(date2[0], date2[1], date2[2], date2[3], date2[4])
    diff = dt2 - dt1
    return diff.total_seconds() / (60 * 60 * 24)


def get_times(data):
    # Define the reference date as opposition 0
    ref_date = data[0]
    ref_date = [ref_date[0], ref_date[1], ref_date[2], ref_date[3], ref_date[4]]
    
    times = []
    for i in range(len(data)):
        current_date = [data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4]]
        days = days_between_dates(ref_date, current_date)
        times.append(days)
    
    return times


def get_oppositions(data):
    longitudes = []
    for row in data:
        zodiac_index = row[5]
        degree = row[6]
        minute = row[7]
        second = row[8]
        longitude = zodiac_index * 30 + degree + minute / 60 + second / 3600
        longitudes.append(longitude)
    return longitudes


def calculate_equant_longitude_of_opp(times, z, s):
    T = 687
    equant_longitude = [z]
    for time in times[1:]:
        #print(f"z: {z}, time: {time}")
        equant_longitude.append((z +  time * s)%360)
    return equant_longitude


def cartesian_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.degrees(np.arctan2(y, x))
    #theta = (theta + 360) % 360  # Ensure theta is in [0, 360)
    return r, theta


def polar_to_cartesian(r, theta):
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    return x, y


def slope_of_line(pt1, pt2):  # Needs cartesian coordinates
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 == x1:
        return float('inf')
    print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
    theta = (y2 - y1) / (x2 - x1)
    return theta


def angle_btw_lines(line1_pt, line2_pt): # Needs cartesian coordinates
    l1_p1, l1_p2 = line1_pt
    l2_p1, l2_p2 = line2_pt

    # Calculate direction vectors
    v1 = np.array([l1_p2[0] - l1_p1[0], l1_p2[1] - l1_p1[1]])
    v2 = np.array([l2_p2[0] - l2_p1[0], l2_p2[1] - l2_p1[1]])

    # Calculate the angle between the vectors using the atan2 function
    angle = np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))

    # Ensure the angle is in the range [0, 360)
    angle = angle % 360

    return angle


def point_of_int_btw_equant_longitude_and_orbit(equant_longitudes, r, c, e1, e2):
    points = []
    e_x, e_y = polar_to_cartesian(e1, e2)  # Equant point
    h, k = polar_to_cartesian(1, c)  # Center of the orbit

    for angle in equant_longitudes:
        # Calculate the slope and y-intercept of the line from equant point
        if abs(angle - 90) < 1e-6 or abs(angle - 270) < 1e-6:
            # Near-vertical line case
            x1 = x2 = e_x
            y1 = k + np.sqrt(r**2 - (x1 - h)**2)
            y2 = k - np.sqrt(r**2 - (x1 - h)**2)
        else:
            slope = np.tan(np.radians(angle))
            b = e_y - slope * e_x
            
            # Calculate intersection points using the quadratic formula
            A = 1 + slope**2
            B = 2 * (slope * (b - k) - h)
            C = h**2 + (b - k)**2 - r**2
            discriminant = B**2 - 4*A*C
            
            if discriminant < 0:
                #print(f"No intersection for equant longitude {angle}")
                continue  # No intersection
            elif abs(discriminant) < 1e-6:
                # One intersection point
                x1 = x2 = -B / (2 * A)
                y1 = y2 = slope * x1 + b
            else:
                # Two intersection points
                x1 = (-B + np.sqrt(discriminant)) / (2 * A)
                y1 = slope * x1 + b
                x2 = (-B - np.sqrt(discriminant)) / (2 * A)
                y2 = slope * x2 + b

        # Convert both points to polar coordinates
        p1_r, p1_theta = cartesian_to_polar(x1 - h, y1 - k)
        p2_r, p2_theta = cartesian_to_polar(x2 - h, y2 - k)
        
        # Calculate the absolute difference between the points' theta and equant longitude
        p1_diff = abs(p1_theta - angle)
        p2_diff = abs(p2_theta - angle)
        
        # Ensure the differences are within the [0, 360) range
        p1_diff = min(p1_diff, 360 - p1_diff)
        p2_diff = min(p2_diff, 360 - p2_diff)
        
        # Choose the point whose theta is closest to the equant longitude
        if p1_diff < p2_diff:
            points.append((x1, y1))
        else:
            points.append((x2, y2))

    return points 


def point_of_int_btw_sun_longitude_and_orbit(longitudes, r, c):
    points = []
    s_x1, s_y1 = 0, 0  # Sun at origin
    h, k = polar_to_cartesian(1, c)  # Center of the orbit
    
    for angle in longitudes:
        slope = np.tan(np.radians(angle))
        b = s_y1 - slope * s_x1
        A = 1 + slope ** 2
        B = 2 * slope * (b - k) - 2 * h
        C = h ** 2 + (b - k) ** 2 - r ** 2
        discriminant = B ** 2 - 4 * A * C

        if discriminant < 0:
            continue
        
        elif abs(discriminant) < 1e-6:
            x1 = -B / (2 * A)
            y1 = slope * x1 + b
            r1, theta1 = cartesian_to_polar(x1, y1)
            if abs(theta1 - angle) < 0.0001:
                points.append((x1, y1))

        else:
            x1 = (-B + np.sqrt(discriminant)) / (2 * A)
            y1 = slope * x1 + b
            x2 = (-B - np.sqrt(discriminant)) / (2 * A)
            y2 = slope * x2 + b

            # Convert both intersection points to polar coordinates
            r1, theta1 = cartesian_to_polar(x1, y1)
            r2, theta2 = cartesian_to_polar(x2, y2)
            p1_diff = abs(theta1 - angle)
            p2_diff = abs(theta2 - angle)
            p1_diff = min(p1_diff, 360 - p1_diff)
            p2_diff = min(p2_diff, 360 - p2_diff)

            # Choose the point whose polar angle is closest to the given longitude
            if p1_diff < p2_diff:
                points.append((x1, y1))
            else:
                points.append((x2, y2))

    return points

# Returns the error of largest magnitude, with its sign
def MarsEquantModel(c,e1,e2,z,r,s,times,oppositions):
    errors = []
    equant_longitudes = calculate_equant_longitude_of_opp(times, z, s)
    intersection_points_equant = point_of_int_btw_equant_longitude_and_orbit(equant_longitudes, r, c, e1, e2)
    intersection_points_sun = point_of_int_btw_sun_longitude_and_orbit(oppositions, r, c)
    for eq_point, sun_point in zip(intersection_points_equant, intersection_points_sun):
        # error for a pair of corresponding points is just the difference in their angles
        error = (-cartesian_to_polar(*eq_point)[1] + cartesian_to_polar(*sun_point)[1])
        errors.append(error)
    # maxError is the error of largest magnitude
    maxError = max(errors, key=abs) 
    return errors, abs(maxError)


def grid_search(c_range, e1_range, e2_range, z_range, r_range, s_range, times, oppositions):
    
    # Initializing variables 
    best_c, best_e1, best_e2, best_z, best_r, best_s = None, None, None, None, None, None
    min_maxError = float('inf')
    best_errors = []
    # Iterate over the parameter combinations
    for c in c_range:
        for e1 in e1_range:
            for e2 in e2_range:
                for z in z_range:
                    for r in r_range:
                        for s in s_range:
                            
                            errors, maxError = MarsEquantModel(c, e1, e2, z, r, s, times, oppositions)

                            # Checking if the current combination gives a smaller maximum error
                            if abs(maxError) < abs(min_maxError):
                                min_maxError = maxError 
                                best_c, best_e1, best_e2, best_z, best_r, best_s = c, e1, e2, z, r, s
                                best_errors = errors
                            
        
    # Return the best parameters and the associated errors
    return best_c, best_e1, best_e2, best_z, best_r, best_s, best_errors, min_maxError


def objective_function(params, times, oppositions):
    c, e1, e2, z, r, s = params
    errors, maxError = MarsEquantModel(c, e1, e2, z, r, s, times, oppositions)
    maxError = 0
    for error in errors:
        maxError = maxError + error**2
    return abs(maxError)


def bestMarsOrbitParams(times, oppositions):
    # Step 1: Broad search over the full parameter ranges
    c_range = np.linspace(145, 155, 21)  
    e1_range = np.linspace(1, 2, 21)  
    e2_range = np.linspace(145, 155, 21)
    z_range = np.linspace(50, 60, 11)  
    r_range = np.linspace(8, 9, 11)  
    s_range = [360/687]

    # Performing the broad search
    best_c, best_e1, best_e2, best_z, best_r, best_s, best_errors, min_maxError = grid_search(
        c_range, e1_range, e2_range, z_range, r_range, s_range, times, oppositions)
    

    # Step 2: Fine search in the neighborhood of the best parameters
    initial_guess = [best_c, best_e1, best_e2, best_z, best_r, best_s]

    bounds = [
        (145, 155),  # bounds for c
        (1, 2.0),  # bounds for e1
        (145, 155),  # bounds for e2
        (50, 60),  # bounds for z
        (5, 9),  # bounds for r
        (0.52408, 0.52408)   # bounds for s
    ]

    # Use scipy.optimize.minimize for the fine search
    result = minimize(
        objective_function,     
        initial_guess,          
        args=(times, oppositions), 
        bounds=bounds,          
        method='L-BFGS-B'       
    )

    best_c, best_e1, best_e2, best_z, best_r, best_s = result.x

    best_errors, min_maxError = MarsEquantModel(best_c, best_e1, best_e2, best_z, best_r, best_s, times, oppositions)

    return best_c, best_e1, best_e2, best_z, best_r, best_s, best_errors, min_maxError


def plot_orbit_and_points(e1, e2, r, c, z, equant_longitudes, oppositions, intersection_points_equant, intersection_points_sun):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set axis limits
    ax.set_xlim([-r - 2, r + 2])
    ax.set_ylim([-r - 2, r + 2])

    # Convert the center of the orbit (polar) to Cartesian
    center_x, center_y = polar_to_cartesian(1, c)
    
    # Plot the orbit (circle with center at (1, c) in polar coordinates and radius r)
    circle = plt.Circle((center_x, center_y), r, color='b', fill=False, label="Orbit")
    ax.add_artist(circle)

    # Plot the Sun at (0, 0)
    ax.scatter(0, 0, color='orange', label="Sun (0,0)", s=100)

    # Convert and plot the equant at (e1, e2)
    equant_x, equant_y = polar_to_cartesian(e1, e2)
    ax.scatter(equant_x, equant_y, color='green', label=f"Equant ({e1},{e2})", s=100)

    # Plot equant longitudes as lines from the equant point
    for angle in equant_longitudes:
        x_end, y_end = polar_to_cartesian(100, angle)  # Extend the line far for visualization
        ax.plot([equant_x, x_end], [equant_y, y_end], color='green', linestyle='--')

    # Plot Sun longitudes as lines from the Sun
    for angle in oppositions:
        x_end, y_end = polar_to_cartesian(100, angle)  # Extend the line far for visualization
        ax.plot([0, x_end], [0, y_end], color='orange', linestyle='--')

    # Plot points of intersection for equant and Sun longitudes with the orbit
    intersection_x_equant, intersection_y_equant = zip(*intersection_points_equant)
    ax.scatter(intersection_x_equant, intersection_y_equant, color='red', label="Intersection (Equant)", s=50)
    
    intersection_x_sun, intersection_y_sun = zip(*intersection_points_sun)
    ax.scatter(intersection_x_sun, intersection_y_sun, color='purple', label="Intersection (Sun)", s=50)

    # Labels and legends
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Orbit, Equant, Sun, and Intersection Points")
    ax.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()



################################################################################################################################


if __name__ == "__main__":

    # Import oppositions data from the CSV file provided
    data = np.genfromtxt(
        "/Users/adityamanjunatha/Library/CloudStorage/OneDrive-IndianInstituteofScience/IISc Semester/5th Semester/Intro to Scalable Systems/Assignments/ds221-2024-main/A1/01_data_mars_opposition_updated.csv",
        delimiter=",",
        skip_header=True,
        dtype="int",
    )

    # Extract times from the data in terms of number of days.
    # "times" is a numpy array of length 12. The first time is the reference
    # time and is taken to be "zero". That is times[0] = 0.0
    times = get_times(data)
    assert len(times) == 12, "times array is not of length 12"

    # Extract angles from the data in degrees. "oppositions" is
    # a numpy array of length 12.
    oppositions = get_oppositions(data)
    assert len(oppositions) == 12, "oppositions array is not of length 12"

    # Call the top level function for optimization
    # The angles are all in degrees
    c= 151.7911041846529 
    r= 8.355667572033346 
    e1= 1.5550014730816122 
    e2= 149.14446333762618 
    z= 55.81617637721456 
    s= 0.52408
    errors, maxError = MarsEquantModel(c,e1,e2,z,r,s,times, oppositions)
    print("Max error: ", maxError)
    print("Errors: ", errors)

"""
    assert max(list(map(abs, errors))) == maxError, "maxError is not computed properly!"
    print(
        "Fit parameters: r = {:.4f}, s = {:.4f}, c = {:.4f}, e1 = {:.4f}, e2 = {:.4f}, z = {:.4f}".format(
            r, s, c, e1, e2, z
        )
    )
    print("The maximum angular error = {:2.4f}".format(maxError))
"""