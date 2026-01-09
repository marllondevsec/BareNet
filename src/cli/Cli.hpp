#pragma once

#include "../filters/FilterManager.hpp"
#include <string>

class Cli {
public:
    Cli();
    void run();

private:
    bool running = true;
    FilterManager filters;

    void showMainMenu();
    void handleMainInput(const std::string& input);

    void filtersMenu();
    void searchScreen();
};
