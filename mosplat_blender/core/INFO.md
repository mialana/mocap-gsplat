# Core Module

This module is reserved for Blender Python API Core logic.

Specifically, it implements these `bpy_struct` classes (`bpy_struct` is the base class for all classes in `bpy.types` module):

### Multiple Instances

1.  [`bpy.types.Operator`](https://docs.blender.org/api/current/bpy.types.Operator.html)
2.  [`bpy.types.Panel`](https://docs.blender.org/api/current/bpy.types.Panel.html)

These types have a hierarchal class tree using mixins and the base blender type.

For more information about the implementation and usefulness of the mixin classes, see [mixins.py](../infrastructure/mixins.py).

### Global Singleton Instance

1.  [`bpy.types.PropertyGroup`](https://docs.blender.org/api/current/bpy.types.PropertyGroup.html#bpy.types.PropertyGroup)
2.  [`bpy.type.AddonPreferences`](https://docs.blender.org/api/current/bpy.types.AddonPreferences.html#bpy.types.AddonPreferences)

These classes have a much simpler inheritance, as they are singletons. But they do still inherit from the log class mixin, so they have class-aware logging capabilities.

```txt
log class mixin <- instance class w/ blender inheritance
```
