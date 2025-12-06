from . import install

bl_info = {
    "name": "VGGT Blender",
    "blender": (4, 2, 0),
    "category": "Pipeline",
}
def register():
    print("Hello World")
    install.main()

def unregister():
    print("Goodbye World")