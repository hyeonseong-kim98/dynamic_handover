#include "dyn_ho_ctl/rg_gripper_client.hpp"

RGGripperClient::RGGripperClient(std::string gripper_type, const std::string& ip, int port) {
    mb = modbus_new_tcp(ip.c_str(), port);
    if (!mb) {
        std::cerr << "Failed to create Modbus TCP context." << std::endl;
        return;
    }
    modbus_set_slave(mb, 65);

    if (gripper_type == "rg2") {uint16_t reg;
        modbus_read_registers(mb, 268, 1, &reg);
        max_width = 1100;
        max_force = 400;
    } else if (gripper_type == "rg6") {
        max_width = 1600;
        max_force = 1200;
    } else {
        std::cerr << "Please specify either 'rg2' or 'rg6'." << std::endl;
    }
    gripper = gripper_type;
}

RGGripperClient::~RGGripperClient() {
    modbus_free(mb);
}

bool RGGripperClient::openConnection() {
    return modbus_connect(mb) == 0;
}

void RGGripperClient::closeConnection() {
    modbus_close(mb);
}

float RGGripperClient::getFingertipOffset() {
    uint16_t reg;
    modbus_read_registers(mb, 258, 1, &reg);
    return reg / 10.0;
}

float RGGripperClient::getWidth() {
    uint16_t reg;
    modbus_read_registers(mb, 267, 1, &reg);
    return reg / 10.0;
}

std::vector<int> RGGripperClient::getStatus() {
    uint16_t reg;
    modbus_read_registers(mb, 268, 1, &reg);
    
    std::bitset<16> status_bits(reg);
    std::string status = status_bits.to_string();
    
    std::vector<int> status_list(7, 0);

    if (status[15] == '1') {
        std::cout << "A motion is ongoing so new commands are not accepted." << std::endl;
        status_list[0] = 1;
    }
    if (status[14] == '1') {
        std::cout << "An internal- or external grip is detected." << std::endl;
        status_list[1] = 1;
    }
    if (status[13] == '1') {
        std::cout << "Safety switch 1 is pushed." << std::endl;
        status_list[2] = 1;
    }
    if (status[12] == '1') {
        std::cout << "Safety circuit 1 is activated so it will not move." << std::endl;
        status_list[3] = 1;
    }
    if (status[11] == '1') {
        std::cout << "Safety switch 2 is pushed." << std::endl;
        status_list[4] = 1;
    }
    if (status[10] == '1') {
        std::cout << "Safety circuit 2 is activated so it will not move." << std::endl;
        status_list[5] = 1;
    }
    if (status[9] == '1') {
        std::cout << "Any of the safety switch is pushed." << std::endl;
        status_list[6] = 1;
    }

    return status_list;
}


float RGGripperClient::getWidthWithOffset() {
    uint16_t reg;
    modbus_read_registers(mb, 275, 1, &reg);
    return reg / 10.0;
}

void RGGripperClient::setControlMode(int command) {
    openConnection();
    modbus_write_register(mb, 2, command);
    closeConnection();
}

void RGGripperClient::setTargetForce(int force_val) {
    openConnection();
    modbus_write_register(mb, 0, force_val);
    closeConnection();
}

void RGGripperClient::setTargetWidth(int width_val) {
    openConnection();
    modbus_write_register(mb, 1, width_val);
    closeConnection();
}

void RGGripperClient::closeGripper(int force) {
    openConnection();
    std::cout << "Start closing gripper." << std::endl;
    std::vector<uint16_t> params = {static_cast<uint16_t>(force), 0, 16};
    while (getStatus()[0]); // Wait until not busy
    modbus_write_registers(mb, 0, 3, params.data());
    closeConnection();
}

void RGGripperClient::openGripper(int force) {
    openConnection();
    std::cout << "Start opening gripper." << std::endl;
    std::vector<uint16_t> params = {static_cast<uint16_t>(force), static_cast<uint16_t>(max_width), 16};
    while (getStatus()[0]); // Wait until not busy
    modbus_write_registers(mb, 0, 3, params.data());
    closeConnection();
}

void RGGripperClient::moveGripper(int width, int force) {
    openConnection();
    std::cout << "Start moving gripper." << std::endl;
    std::vector<uint16_t> params = {static_cast<uint16_t>(force), static_cast<uint16_t>(width), 16};
    while (getStatus()[0]); // Wait until not busy
    modbus_write_registers(mb, 0, 3, params.data());
    closeConnection();
}