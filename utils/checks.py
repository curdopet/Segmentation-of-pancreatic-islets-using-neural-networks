def is_image(file_name: str) -> bool:
    return file_name.endswith(".png") or file_name.endswith(".jpg") or \
           file_name.endswith(".bmp") or file_name.endswith(".tif")
