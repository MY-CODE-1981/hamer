$ catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE='/home/initial/miniconda3/envs/hamer/bin/python'

source /home/initial/workspace/hamer_ws/devel/setup.bash
python demo_ycb_renderer.py

source /home/initial/workspace/hamer_ws/devel/setup.bash
python demo_ycb_renderer_ros.py
