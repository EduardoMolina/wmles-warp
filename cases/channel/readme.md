# Turbulent 2D Channel grids

The domain size is taken as (12.8⁢h,2⁢h,4.8⁢h), which is smaller than in the DNS by about factors of 2 in the wall-parallel directions. This is done to reduce the computational cost, and will not affect the assessment of the wall-modeling (provided we interpret the results correctly, e.g., accepting differences in the wake region).

The grids need to be extruded in the 3D dimension by 49 ($\delta_z = 0.1h$)

https://wmles.umd.edu/validation/channel-flow-at-re5200/channel-grids/

```python
import plot3d
blocks = plot3d.read_plot3D('channel_ar1_1dwm_129x33.p3d')
```