import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_colmap_images(file_path):
    """
    Parses COLMAP images.txt file to extract camera poses and convert them to world coordinates.
    
    Args:
        file_path (str): Path to the COLMAP images.txt file.
    
    Returns:
        numpy.ndarray: An (N x 4 x 4) NumPy array containing camera poses in world coordinates.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    camera_poses = []
    for line in lines:
        # Skip comments and empty lines
        if line.startswith('#') or not line.strip():
            continue

        # COLMAP image format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        elements = line.split()
        if len(elements) < 8:  # Ensure it's a valid image line
            continue

        # Extract quaternion (QW, QX, QY, QZ) and translation (TX, TY, TZ)
        qw, qx, qy, qz = map(float, elements[1:5])
        tx, ty, tz = map(float, elements[5:8])

        # Convert quaternion to rotation matrix
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

        # Create 4x4 transformation matrix (world to camera)
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = [tx, ty, tz]

        # Invert the transformation to get camera to world
        camera_to_world = np.linalg.inv(world_to_camera)

        # Store the camera-to-world pose
        camera_poses.append(camera_to_world)

    # Convert to NumPy array
    camera_poses = np.array(camera_poses)

    return camera_poses

# Example usage
file_path = "/home/e/eez095/dexycb_data/20200813-subject-02/20200813_155021/12_frame/sparse/0/images.txt"  # Replace with your images.txt file path
camera_poses = parse_colmap_images(file_path)

print("Camera Poses (in world coordinates):")
print(camera_poses)
print(f"Shape: {camera_poses.shape}")


gt = np.array([
        [[-0.893451988697052, -0.03789233788847923, -0.4475576877593994, 0.6386182308197021],
        [-0.3177464008331299, 0.7575904130935669, 0.5701702237129211, -0.5989090204238892],
        [0.3174603581428528, 0.6516294479370117, -0.688910722732544, 1.391905665397644],
        [0, 0, 0, 1]],

        [[-0.2367057204246521, 0.26019808650016785, 0.9360916018486023, -0.6338921785354614],
        [0.7574113011360168, 0.6528606414794922, 0.010052978061139584, -0.08515024185180664],
        [-0.6085217595100403, 0.7113859057426453, -0.35161277651786804, 1.0669708251953125],
        [0, 0, 0, 1]],

        [[1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0, 0, 0, 1]],

        [[0.5951622619388115, 0.6853450488717701, -0.4196236956498394, 0.7148191928863525],
        [-0.6514453394194257, 0.7172168040858102, 0.2474247879369531, -0.3177867531776428],
        [0.47053251929339285, 0.12610400439275565, 0.8733252134019291, 0.18353232741355896],
        [0, 0, 0, 1]],

        [[-0.4766446146572601, 0.75390154568288, -0.4521530390644259, 0.41736653447151184],
        [-0.6333187685169396, 0.06223099033979218, 0.7713848853105145, -0.6476160287857056],
        [0.609686188758134, 0.6540334572927389, 0.447797932084649, 0.40139156579971313],
        [0, 0, 0, 1]],

        [[0.4486923103476053, -0.6708688872680653, 0.5904321694577928, -0.1706404834985733],
        [0.6349333263878278, 0.7042357829222984, 0.3176659142777285, -0.46829313039779663],
        [-0.6289156395551123, 0.2323508083642598, 0.7419421946320743, 0.28708332777023315],
        [0, 0, 0, 1]],

        [[-0.2469573303563285, -0.5030590752630304, 0.8282171477208013, -0.5300791263580322],
        [0.6828783108268728, 0.516055556015601, 0.5170724085731763, -0.5804263353347778],
        [-0.6875240283695634, 0.6932663484555798, 0.21608442913258327, 0.7049697637557983],
        [0, 0, 0, 1]],

        [[-0.9286197279370463, 0.28499809028787526, -0.23757417666571096, 0.35575658082962036],
        [-0.19175885377662133, 0.17951273203290213, 0.9648853408754638, -1.0083527565002441],
        [0.31763806901000186, 0.9415685145385586, -0.1120481572595004, 0.9933208227157593],
        [0, 0, 0, 1]]
        ]
)
def compare_numpy_arrays(array1, array2):
    """
    Compare two NumPy arrays and display where they differ if they are not identical.

    Args:
        array1 (numpy.ndarray): The first NumPy array.
        array2 (numpy.ndarray): The second NumPy array.

    Returns:
        bool: True if the arrays are identical, False otherwise.
    """
    # Check if arrays are identical
    if np.array_equal(array1, array2):
        print("The arrays are identical.")
        return True

    # Find differences
    if array1.shape != array2.shape:
        print(f"The arrays have different shapes: {array1.shape} vs {array2.shape}")
        return False

    # Find indices where arrays differ
    difference_mask = array1 != array2
    differing_indices = np.argwhere(difference_mask)

    # Display differing elements
    print("The arrays are different. Differing elements:")
    for index in differing_indices:
        index_tuple = tuple(index)
        print(f"Index {index_tuple}: array1={array1[index_tuple]}, array2={array2[index_tuple]}")
    
    return False

gt = np.round(gt,8)
camera_poses = np.round(camera_poses, 8)
compare_numpy_arrays(gt,camera_poses)
