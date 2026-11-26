import time
from rg_gripper import RGGripperClient

rg = RGGripperClient(gripper='rg2', ip="192.168.0.4", port="502")

if __name__ == '__main__':
    rg.close_gripper(force=400)
    time.sleep(3)
    print(rg.get_width_with_offset())