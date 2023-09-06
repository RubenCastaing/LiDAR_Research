import open3d as o3d

filename = r"/csse/users/rca106/Desktop/Semester 2 2023/Blender/Uni.obj"

# Read the PLY file
ply = o3d.io.read_triangle_mesh(filename)

# Visualization
o3d.visualization.draw_geometries([ply])
