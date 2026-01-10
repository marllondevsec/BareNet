#include <iostream>
#include "Banner.hpp"
#include "Colors.hpp"

void Banner::render()
{
    using std::cout;

    cout << Colors::RED << Colors::BOLD;

    cout << R"(

██████╗  █████╗ ██████╗ ███████╗███╗   ██╗███████╗████████╗
██╔══██╗██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝╚══██╔══╝
██████╔╝███████║██████╔╝█████╗  ██╔██╗ ██║█████╗     ██║   
██╔══██╗██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║██╔══╝     ██║   
██████╔╝██║  ██║██║  ██║███████╗██║ ╚████║███████╗   ██║   
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝   

)";
    cout << Colors::RESET;

    cout << "   BareNet  v0.1  |  navegador passivo de OSINT\n";
    cout << "   Linux-first • CLI • modular • extensível\n\n";
}
