#include <iostream>
#include <unistd.h>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include <algorithm>

int main(int argc, char** argv) {
 //const int numthreads = argc > 1 ? atoi(argv[1]) : 0;
 /* std::vector< std::thread > threads;
 
        for(int i = 0; i != numthreads; ++i) 
           threads.push_back(std::move(std::thread([]{
                std::this_thread::sleep_for(std::chrono::seconds(60));
           })));*/
 std::thread t = std::thread([]() {
        const pid_t pid = getpid();
        std::ostringstream oss;
        oss << "ps uH p " << pid;
        FILE* f = popen(oss.str().c_str(), "r");
        std::string psout;
        std::vector< char > buffer(0x10000);
        size_t count = fread(&buffer[0], sizeof(char), buffer.size(), f);
        while(count) {
                buffer[count] = '\0';
                psout += &buffer[0];
                count = fread(&buffer[0], sizeof(char), buffer.size(), f);
        }
        std::cout << psout << std::endl;
        pclose(f);
        });
t.join();
//std::for_each(threads.begin(), threads.end(), [](std::thread& t) {
//                                              t.join();
//                                            });
return 0;
}
