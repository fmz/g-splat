import pycolmap



def reconstruction():

    reconstruction = pycolmap._core.Reconstruction("data/db/drjohnson/sparse" )
    for camera_id, camera in reconstruction.cameras.items():
        print(f"Camera ID: {camera_id}, Model: {camera.model}, Parameters: {camera.params}")

    for image_id, image in reconstruction.images.items():
        print(f"Image ID: {image_id}, Name: {image.name}, Qvec: {image.qvec}, Tvec: {image.tvec}")

    for point_id, point in reconstruction.points3D.items():
        print(f"Point ID: {point_id}, XYZ: {point.xyz}, Color: {point.color}")
    return


if __name__=="__main__":
    reconstruction()

