import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

CAM_ANGLE_V = np.radians(27.0)
CAM_ANGLE_H = np.radians(39.6)
GRID_START_LOC = np.array([-1.5,-1.1,-2])
GRID_SIZE = np.array([60,44,80]) # num voxels in grid
VOXEL_SIZE = 0.05 # size of voxel
SIMILARITY_THRESHOLD = 0.23 # threshold needed for colors to be considered consistent
FOLDER_PATH = "figure/"

# stores the info of a camera, as well as its associated photo
class Camera:
  def __init__(self, loc, rot, view):
    self.u = eulerXYZ(np.array([[1], [0], [0], [1]]), rot)
    self.v = eulerXYZ(np.array([[0], [1], [0], [1]]), rot)
    self.w = eulerXYZ(np.array([[0], [0], [1], [1]]), rot)
    # matrix for converting coords to camera space
    self.mat = np.matmul(np.array([[self.u[0][0], self.u[1][0], self.u[2][0], 0],
                                [self.v[0][0], self.v[1][0], self.v[2][0], 0],
                                [self.w[0][0], self.w[1][0], self.w[2][0], 0],
                                [0, 0, 0, 1]]),
                         np.array([[1, 0, 0, -loc[0]],
                                   [0, 1, 0, -loc[1]],
                                   [0, 0, 1, -loc[2]],
                                   [0, 0, 0, 1]]))
    self.loc = loc
    self.view = view

# rotates a vector with euler angles
def eulerXYZ(vec, rot):
  x_rotated = np.matmul(np.array([[1, 0, 0, 0],
                                 [0, np.cos(rot[0]), -np.sin(rot[0]), 0],
                                 [0, np.sin(rot[0]), np.cos(rot[0]), 0],
                                 [0, 0, 0, 1]]), vec)
  y_rotated = np.matmul(np.array([[np.cos(rot[1]), 0, np.sin(rot[1]), 0],
                                 [0, 1, 0, 0],
                                 [-np.sin(rot[1]), 0, np.cos(rot[1]), 0],
                                 [0, 0, 0, 1]]), x_rotated)
  z_rotated = np.matmul(np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0, 0],
                                 [np.sin(rot[2]), np.cos(rot[2]), 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]), y_rotated)
  return z_rotated

# sweeps through the volume in a given direction, carving plane by plane
def plane_sweep(cams: list, axis: int, is_negative: bool, volume: o3d.geometry.VoxelGrid, 
                grid_colors: np.array, voxels_to_remove: list, carved_voxels: np.array):
  dir_vec = np.array([0, 0, 0, 1])
  # direction is determined by axis and is_negative
  dir_vec[axis] = (-1) ** int(is_negative)

  cams_in_dir = get_cams_in_dir(cams, dir_vec)

  if len(cams_in_dir) == 0:
    return voxels_to_remove, volume

  # rearranges the grid array so that the for loop will iterate along the given axis
  order = np.array([axis, (axis + 1) % 3, (axis + 2) % 3])
  grid_rearranged = GRID_SIZE[order]
  # tracks which voxels in the sweep plane are consistent
  voxels_covered = np.full((grid_rearranged[1], grid_rearranged[2]), False)

  # iterates backwards if the direction is negative
  for i in range((grid_rearranged[0] - 1) * int(is_negative),
                 ((grid_rearranged[0] + 1) * int(not is_negative)) - 1,
                 (-1) ** int(is_negative)):
    for j in range(grid_rearranged[1]):
      for k in range(grid_rearranged[2]):
        # gets normal index for accessing voxels
        index = np.array([i, j, k])[order][order]

        # only checks for consistency if the voxel has not been carved 
        # or has the same jk-coordinate as a consistent voxel
        if not voxels_covered[j][k] and not carved_voxels[index[0],index[1],index[2]]:
          voxel_loc = np.array([np.append(volume.get_voxel_center_coordinate(index), 1)]).transpose()
          colors = get_cam_colors(cams_in_dir, voxel_loc)

          if colors:
            if consist(colors):
              grid_colors[index[0], index[1], index[2], 3 * int(is_negative) + axis] = colors[0]
              voxels_covered[j][k] = True
            else:
              voxels_to_remove.append(index)
              carved_voxels[index[0],index[1],index[2]] = True

# returns the cameras facing in the same direction as dir_vec
def get_cams_in_dir(cams: list, dir_vec: np.array, dir_threshhold = 0.1):
  cams_in_dir = []
  for i in range(len(cams)):
    if (np.dot(cams[i].w.transpose(), dir_vec) - 1) < dir_threshhold:
      cams_in_dir.append(cams[i])
  return cams_in_dir

# gets the pixel color that a voxel projects to for each camera
def get_cam_colors(cams: list, voxel_loc: np.array):
  colors = []
  for i in range(len(cams)):
    voxel_cam_loc = np.matmul(cams[i].mat, voxel_loc)
    x_coord = (voxel_cam_loc[0][0] / voxel_cam_loc[2][0]) / (2 * np.tan(CAM_ANGLE_H / 2)) + 0.5
    y_coord = (voxel_cam_loc[1][0] / voxel_cam_loc[2][0]) / (2 * np.tan(CAM_ANGLE_V / 2)) + 0.5

    if (x_coord >= 0 and y_coord >= 0) and (x_coord < 1 and y_coord < 1):
      x_pixel = int(np.floor(x_coord * len(cams[i].view[0])))
      y_pixel = int(np.floor(y_coord * len(cams[i].view)))
      pixel_color = np.array([cams[i].view[y_pixel, x_pixel, 0],
                            cams[i].view[y_pixel, x_pixel, 1],
                            cams[i].view[y_pixel, x_pixel, 2]]).astype(np.float64)
      colors.append(pixel_color)
  return colors

# determines if the given colors are similar enough to be consistent
def consist(colors: list):
  if np.std(np.transpose(colors)[0]) + np.std(np.transpose(colors)[1]) + np.std(np.transpose(colors)[2]) > SIMILARITY_THRESHOLD or not colors:
    return False
  return True

# processes cam info in cam_info.txt to create cameras
cams = []
f = open(FOLDER_PATH + "cam_info.txt", "r")
for line in f:
  info = line.split()
  loc = info[0].split(",")
  rot = info[1].split(",")
  img = plt.imread(FOLDER_PATH + info[2])
  cams.append(Camera(np.array([float(loc[0]), float(loc[1]), float(loc[2])]),
              np.array([np.radians(float(rot[0])),
                        np.radians(float(rot[1])),
                        np.radians(float(rot[2]))]),
              img))
f.close

# initializes voxel grid
volume = o3d.geometry.VoxelGrid.create_dense(
    GRID_START_LOC, np.array([0, 0, 0]), VOXEL_SIZE,
    GRID_SIZE[0]*VOXEL_SIZE, GRID_SIZE[1]*VOXEL_SIZE, GRID_SIZE[2]*VOXEL_SIZE)
carved_voxels = np.full((GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2]), False)

while True:
  # stores consistent colors from each plane sweep
  grid_colors = np.full((GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2], 6), None)
  # stores voxels to be carved
  voxels_to_remove = []

  # plane sweeps in each principal direction
  for i in range(3):
    for j in range(2):
      plane_sweep(cams, i, bool(j), volume, grid_colors, voxels_to_remove, carved_voxels)

  for voxel in voxels_to_remove:
    volume.remove_voxel(voxel)

  for voxel in volume.get_voxels():
    index = voxel.grid_index
    voxel_colors = []
    for color in grid_colors[index[0], index[1], index[2]]:
        if hasattr(color, "__len__"):
            voxel_colors.append(color)

    # carves a voxel if colors from plane sweeps are inconsistent
    if len(voxel_colors) > 0:
      volume.remove_voxel(index)
      if consist(voxel_colors):
        volume.add_voxel(o3d.cpu.pybind.geometry.Voxel(index, voxel_colors[0]))
  
  # stops if no voxels were removed
  if len(voxels_to_remove) == 0:
    break

o3d.visualization.draw_geometries([volume])