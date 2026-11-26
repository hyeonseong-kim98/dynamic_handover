import time
from rg_gripper import RGGripperClient

rg = RGGripperClient(gripper='rg2', ip="192.168.0.4", port="502")

if __name__ == '__main__':
    print(rg.get_width_with_offset())
    rg.open_gripper(force=200)
    time.sleep(3)
    rg.close_gripper(force=200)
    time.sleep(3)
    rg.move_gripper(width=400,force=200)
    time.sleep(3)
    print(rg.get_width_with_offset())
    