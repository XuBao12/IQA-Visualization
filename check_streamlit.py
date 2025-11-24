import streamlit
import inspect

try:
    import streamlit.elements.image as st_image
    print("streamlit.elements.image exists")
    if hasattr(st_image, 'image_to_url'):
        print("image_to_url found in streamlit.elements.image")
        print(inspect.signature(st_image.image_to_url))
    else:
        print("image_to_url NOT found in streamlit.elements.image")
except ImportError:
    print("streamlit.elements.image does not exist")

try:
    from streamlit.runtime.media_file_manager import MediaFileManager
    print("MediaFileManager exists")
except ImportError:
    print("MediaFileManager does not exist")
