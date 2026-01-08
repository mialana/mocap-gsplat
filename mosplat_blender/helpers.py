import bpy


def resolve_script_file_path(path: str) -> str:
    filepath = bpy.utils.user_resource("SCRIPTS", path=path, create=True)
    return filepath


def get_window_manager() -> bpy.types.WindowManager:
    return bpy.context.window_manager


def display_message(message, title="Notification", icon="INFO"):
    def draw(self, context):
        self.layout.label(text=message)

    def show_popup():
        bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)
        return None  # Stops timer

    bpy.app.timers.register(show_popup)
