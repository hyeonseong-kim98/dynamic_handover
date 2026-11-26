#ifndef RG_GRIPPER_CLIENT_H
#define RG_GRIPPER_CLIENT_H

#include <modbus/modbus.h>
#include <iostream>
#include <vector>
#include <bitset>


class RGGripperClient {
private:
    modbus_t *mb;
    std::string gripper;
    int max_width;
    int max_force;

public:
    RGGripperClient(std::string gripper_type, const std::string& ip, int port);
    ~RGGripperClient();

    bool openConnection();
    void closeConnection();

    float getFingertipOffset();
    float getWidth();
    std::vector<int> getStatus();
    float getWidthWithOffset();

    void setControlMode(int command);
    void setTargetForce(int force_val);
    void setTargetWidth(int width_val);
    
    void closeGripper(int force = 200);
    void openGripper(int force = 200);
    void moveGripper(int width, int force = 200);
};

#endif
